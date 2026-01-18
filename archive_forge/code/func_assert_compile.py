from __future__ import annotations
from collections import defaultdict
import contextlib
from copy import copy
from itertools import filterfalse
import re
import sys
import warnings
from . import assertsql
from . import config
from . import engines
from . import mock
from .exclusions import db_spec
from .util import fail
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import types as sqltypes
from .. import util
from ..engine import default
from ..engine import url
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import decorator
def assert_compile(self, clause, result, params=None, checkparams=None, for_executemany=False, check_literal_execute=None, check_post_param=None, dialect=None, checkpositional=None, check_prefetch=None, use_default_dialect=False, allow_dialect_select=False, supports_default_values=True, supports_default_metavalue=True, literal_binds=False, render_postcompile=False, schema_translate_map=None, render_schema_translate=False, default_schema_name=None, from_linting=False, check_param_order=True, use_literal_execute_for_simple_int=False):
    if use_default_dialect:
        dialect = default.DefaultDialect()
        dialect.supports_default_values = supports_default_values
        dialect.supports_default_metavalue = supports_default_metavalue
    elif allow_dialect_select:
        dialect = None
    else:
        if dialect is None:
            dialect = getattr(self, '__dialect__', None)
        if dialect is None:
            dialect = config.db.dialect
        elif dialect == 'default' or dialect == 'default_qmark':
            if dialect == 'default':
                dialect = default.DefaultDialect()
            else:
                dialect = default.DefaultDialect('qmark')
            dialect.supports_default_values = supports_default_values
            dialect.supports_default_metavalue = supports_default_metavalue
        elif dialect == 'default_enhanced':
            dialect = default.StrCompileDialect()
        elif isinstance(dialect, str):
            dialect = url.URL.create(dialect).get_dialect()()
    if default_schema_name:
        dialect.default_schema_name = default_schema_name
    kw = {}
    compile_kwargs = {}
    if schema_translate_map:
        kw['schema_translate_map'] = schema_translate_map
    if params is not None:
        kw['column_keys'] = list(params)
    if literal_binds:
        compile_kwargs['literal_binds'] = True
    if render_postcompile:
        compile_kwargs['render_postcompile'] = True
    if use_literal_execute_for_simple_int:
        compile_kwargs['use_literal_execute_for_simple_int'] = True
    if for_executemany:
        kw['for_executemany'] = True
    if render_schema_translate:
        kw['render_schema_translate'] = True
    if from_linting or getattr(self, 'assert_from_linting', False):
        kw['linting'] = sql.FROM_LINTING
    from sqlalchemy import orm
    if isinstance(clause, orm.Query):
        stmt = clause._statement_20()
        stmt._label_style = LABEL_STYLE_TABLENAME_PLUS_COL
        clause = stmt
    if compile_kwargs:
        kw['compile_kwargs'] = compile_kwargs

    class DontAccess:

        def __getattribute__(self, key):
            raise NotImplementedError('compiler accessed .statement; use compiler.current_executable')

    class CheckCompilerAccess:

        def __init__(self, test_statement):
            self.test_statement = test_statement
            self._annotations = {}
            self.supports_execution = getattr(test_statement, 'supports_execution', False)
            if self.supports_execution:
                self._execution_options = test_statement._execution_options
                if hasattr(test_statement, '_returning'):
                    self._returning = test_statement._returning
                if hasattr(test_statement, '_inline'):
                    self._inline = test_statement._inline
                if hasattr(test_statement, '_return_defaults'):
                    self._return_defaults = test_statement._return_defaults

        @property
        def _variant_mapping(self):
            return self.test_statement._variant_mapping

        def _default_dialect(self):
            return self.test_statement._default_dialect()

        def compile(self, dialect, **kw):
            return self.test_statement.compile.__func__(self, dialect=dialect, **kw)

        def _compiler(self, dialect, **kw):
            return self.test_statement._compiler.__func__(self, dialect, **kw)

        def _compiler_dispatch(self, compiler, **kwargs):
            if hasattr(compiler, 'statement'):
                with mock.patch.object(compiler, 'statement', DontAccess()):
                    return self.test_statement._compiler_dispatch(compiler, **kwargs)
            else:
                return self.test_statement._compiler_dispatch(compiler, **kwargs)
    c = CheckCompilerAccess(clause).compile(dialect=dialect, **kw)
    if isinstance(clause, sqltypes.TypeEngine):
        cache_key_no_warnings = clause._static_cache_key
        if cache_key_no_warnings:
            hash(cache_key_no_warnings)
    else:
        cache_key_no_warnings = clause._generate_cache_key()
        if cache_key_no_warnings:
            hash(cache_key_no_warnings[0])
    param_str = repr(getattr(c, 'params', {}))
    param_str = param_str.encode('utf-8').decode('ascii', 'ignore')
    print(('\nSQL String:\n' + str(c) + param_str).encode('utf-8'))
    cc = re.sub('[\\n\\t]', '', str(c))
    eq_(cc, result, '%r != %r on dialect %r' % (cc, result, dialect))
    if checkparams is not None:
        if render_postcompile:
            expanded_state = c.construct_expanded_state(params, escape_names=False)
            eq_(expanded_state.parameters, checkparams)
        else:
            eq_(c.construct_params(params), checkparams)
    if checkpositional is not None:
        if render_postcompile:
            expanded_state = c.construct_expanded_state(params, escape_names=False)
            eq_(tuple([expanded_state.parameters[x] for x in expanded_state.positiontup]), checkpositional)
        else:
            p = c.construct_params(params, escape_names=False)
            eq_(tuple([p[x] for x in c.positiontup]), checkpositional)
    if check_prefetch is not None:
        eq_(c.prefetch, check_prefetch)
    if check_literal_execute is not None:
        eq_({c.bind_names[b]: b.effective_value for b in c.literal_execute_params}, check_literal_execute)
    if check_post_param is not None:
        eq_({c.bind_names[b]: b.effective_value for b in c.post_compile_params}, check_post_param)
    if check_param_order and getattr(c, 'params', None):

        def get_dialect(paramstyle, positional):
            cp = copy(dialect)
            cp.paramstyle = paramstyle
            cp.positional = positional
            return cp
        pyformat_dialect = get_dialect('pyformat', False)
        pyformat_c = clause.compile(dialect=pyformat_dialect, **kw)
        stmt = re.sub('[\\n\\t]', '', str(pyformat_c))
        qmark_dialect = get_dialect('qmark', True)
        qmark_c = clause.compile(dialect=qmark_dialect, **kw)
        values = list(qmark_c.positiontup)
        escaped = qmark_c.escaped_bind_names
        for post_param in qmark_c.post_compile_params | qmark_c.literal_execute_params:
            name = qmark_c.bind_names[post_param]
            if name in values:
                values = [v for v in values if v != name]
        positions = []
        pos_by_value = defaultdict(list)
        for v in values:
            try:
                if v in pos_by_value:
                    start = pos_by_value[v][-1]
                else:
                    start = 0
                esc = escaped.get(v, v)
                pos = stmt.index('%%(%s)s' % (esc,), start) + 2
                positions.append(pos)
                pos_by_value[v].append(pos)
            except ValueError:
                msg = 'Expected to find bindparam %r in %r' % (v, stmt)
                assert False, msg
        ordered = all((positions[i - 1] < positions[i] for i in range(1, len(positions))))
        expected = [v for _, v in sorted(zip(positions, values))]
        msg = 'Order of parameters %s does not match the order in the statement %s. Statement %r' % (values, expected, stmt)
        is_true(ordered, msg)
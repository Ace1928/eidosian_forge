from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
@classmethod
def orm_pre_session_exec(cls, session, statement, params, execution_options, bind_arguments, is_pre_event):
    load_options, execution_options = QueryContext.default_load_options.from_execution_options('_sa_orm_load_options', {'populate_existing', 'autoflush', 'yield_per', 'identity_token', 'sa_top_level_orm_context'}, execution_options, statement._execution_options)
    if 'sa_top_level_orm_context' in execution_options:
        ctx = execution_options['sa_top_level_orm_context']
        execution_options = ctx.query._execution_options.merge_with(ctx.execution_options, execution_options)
    if not execution_options:
        execution_options = _orm_load_exec_options
    else:
        execution_options = execution_options.union(_orm_load_exec_options)
    if load_options._yield_per:
        execution_options = execution_options.union({'yield_per': load_options._yield_per})
    if getattr(statement._compile_options, '_current_path', None) and len(statement._compile_options._current_path) > 10 and (execution_options.get('compiled_cache', True) is not None):
        execution_options: util.immutabledict[str, Any] = execution_options.union({'compiled_cache': None, '_cache_disable_reason': 'excess depth for ORM loader options'})
    bind_arguments['clause'] = statement
    try:
        plugin_subject = statement._propagate_attrs['plugin_subject']
    except KeyError:
        assert False, "statement had 'orm' plugin but no plugin_subject"
    else:
        if plugin_subject:
            bind_arguments['mapper'] = plugin_subject.mapper
    if not is_pre_event and load_options._autoflush:
        session._autoflush()
    return (statement, execution_options)
from __future__ import annotations
from contextlib import contextmanager
import re
import textwrap
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List  # noqa
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence  # noqa
from typing import Tuple
from typing import Type  # noqa
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.elements import conv
from . import batch
from . import schemaobj
from .. import util
from ..util import sqla_compat
from ..util.compat import formatannotation_fwdref
from ..util.compat import inspect_formatargspec
from ..util.compat import inspect_getfullargspec
from ..util.sqla_compat import _literal_bindparam
@classmethod
def register_operation(cls, name: str, sourcename: Optional[str]=None) -> Callable[[Type[_T]], Type[_T]]:
    """Register a new operation for this class.

        This method is normally used to add new operations
        to the :class:`.Operations` class, and possibly the
        :class:`.BatchOperations` class as well.   All Alembic migration
        operations are implemented via this system, however the system
        is also available as a public API to facilitate adding custom
        operations.

        .. seealso::

            :ref:`operation_plugins`


        """

    def register(op_cls: Type[_T]) -> Type[_T]:
        if sourcename is None:
            fn = getattr(op_cls, name)
            source_name = fn.__name__
        else:
            fn = getattr(op_cls, sourcename)
            source_name = fn.__name__
        spec = inspect_getfullargspec(fn)
        name_args = spec[0]
        assert name_args[0:2] == ['cls', 'operations']
        name_args[0:2] = ['self']
        args = inspect_formatargspec(*spec, formatannotation=formatannotation_fwdref)
        num_defaults = len(spec[3]) if spec[3] else 0
        defaulted_vals: Tuple[Any, ...]
        if num_defaults:
            defaulted_vals = tuple(name_args[0 - num_defaults:])
        else:
            defaulted_vals = ()
        defaulted_vals += tuple(spec[4])
        apply_kw = inspect_formatargspec(name_args + spec[4], spec[1], spec[2], defaulted_vals, formatvalue=lambda x: '=' + x, formatannotation=formatannotation_fwdref)
        args = re.sub('[_]?ForwardRef\\(([\\\'"].+?[\\\'"])\\)', lambda m: m.group(1), args)
        func_text = textwrap.dedent('            def %(name)s%(args)s:\n                %(doc)r\n                return op_cls.%(source_name)s%(apply_kw)s\n            ' % {'name': name, 'source_name': source_name, 'args': args, 'apply_kw': apply_kw, 'doc': fn.__doc__})
        globals_ = dict(globals())
        globals_.update({'op_cls': op_cls})
        lcl: Dict[str, Any] = {}
        exec(func_text, globals_, lcl)
        setattr(cls, name, lcl[name])
        fn.__func__.__doc__ = 'This method is proxied on the :class:`.%s` class, via the :meth:`.%s.%s` method.' % (cls.__name__, cls.__name__, name)
        if hasattr(fn, '_legacy_translations'):
            lcl[name]._legacy_translations = fn._legacy_translations
        return op_cls
    return register
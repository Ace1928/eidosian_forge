from __future__ import annotations
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from . import util as orm_util
from ._typing import insp_is_aliased_class
from ._typing import insp_is_attribute
from ._typing import insp_is_mapper
from ._typing import insp_is_mapper_property
from .attributes import QueryableAttribute
from .base import InspectionAttr
from .interfaces import LoaderOption
from .path_registry import _DEFAULT_TOKEN
from .path_registry import _StrPathToken
from .path_registry import _WILDCARD_TOKEN
from .path_registry import AbstractEntityRegistry
from .path_registry import path_is_property
from .path_registry import PathRegistry
from .path_registry import TokenRegistry
from .util import _orm_full_deannotate
from .util import AliasedInsp
from .. import exc as sa_exc
from .. import inspect
from .. import util
from ..sql import and_
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import traversals
from ..sql import visitors
from ..sql.base import _generative
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Self
def loader_unbound_fn(fn: _FN) -> _FN:
    """decorator that applies docstrings between standalone loader functions
    and the loader methods on :class:`._AbstractLoad`.

    """
    bound_fn = getattr(_AbstractLoad, fn.__name__)
    fn_doc = bound_fn.__doc__
    bound_fn.__doc__ = f'Produce a new :class:`_orm.Load` object with the\n:func:`_orm.{fn.__name__}` option applied.\n\nSee :func:`_orm.{fn.__name__}` for usage examples.\n\n'
    fn.__doc__ = fn_doc
    return fn
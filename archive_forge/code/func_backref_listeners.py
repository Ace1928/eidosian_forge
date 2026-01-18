from __future__ import annotations
import dataclasses
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import collections
from . import exc as orm_exc
from . import interfaces
from ._typing import insp_is_aliased_class
from .base import _DeclarativeMapped
from .base import ATTR_EMPTY
from .base import ATTR_WAS_SET
from .base import CALLABLES_OK
from .base import DEFERRED_HISTORY_LOAD
from .base import INCLUDE_PENDING_MUTATIONS  # noqa
from .base import INIT_OK
from .base import instance_dict as instance_dict
from .base import instance_state as instance_state
from .base import instance_str
from .base import LOAD_AGAINST_COMMITTED
from .base import LoaderCallableStatus
from .base import manager_of_class as manager_of_class
from .base import Mapped as Mapped  # noqa
from .base import NEVER_SET  # noqa
from .base import NO_AUTOFLUSH
from .base import NO_CHANGE  # noqa
from .base import NO_KEY
from .base import NO_RAISE
from .base import NO_VALUE
from .base import NON_PERSISTENT_OK  # noqa
from .base import opt_manager_of_class as opt_manager_of_class
from .base import PASSIVE_CLASS_MISMATCH  # noqa
from .base import PASSIVE_NO_FETCH
from .base import PASSIVE_NO_FETCH_RELATED  # noqa
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import PASSIVE_ONLY_PERSISTENT
from .base import PASSIVE_RETURN_NO_VALUE
from .base import PassiveFlag
from .base import RELATED_OBJECT_OK  # noqa
from .base import SQL_OK  # noqa
from .base import SQLORMExpression
from .base import state_str
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
from ..sql.visitors import _TraverseInternalsType
from ..sql.visitors import InternalTraversal
from ..util.typing import Literal
from ..util.typing import Self
from ..util.typing import TypeGuard
def backref_listeners(attribute: QueryableAttribute[Any], key: str, uselist: bool) -> None:
    """Apply listeners to synchronize a two-way relationship."""
    parent_token = attribute.impl.parent_token
    parent_impl = attribute.impl

    def _acceptable_key_err(child_state, initiator, child_impl):
        raise ValueError('Bidirectional attribute conflict detected: Passing object %s to attribute "%s" triggers a modify event on attribute "%s" via the backref "%s".' % (state_str(child_state), initiator.parent_token, child_impl.parent_token, attribute.impl.parent_token))

    def emit_backref_from_scalar_set_event(state, child, oldchild, initiator, **kw):
        if oldchild is child:
            return child
        if oldchild is not None and oldchild is not PASSIVE_NO_RESULT and (oldchild is not NO_VALUE):
            old_state, old_dict = (instance_state(oldchild), instance_dict(oldchild))
            impl = old_state.manager[key].impl
            if not impl.collection and (not impl.dynamic):
                check_recursive_token = impl._replace_token
            else:
                check_recursive_token = impl._remove_token
            if initiator is not check_recursive_token:
                impl.pop(old_state, old_dict, state.obj(), parent_impl._append_token, passive=PASSIVE_NO_FETCH)
        if child is not None:
            child_state, child_dict = (instance_state(child), instance_dict(child))
            child_impl = child_state.manager[key].impl
            if initiator.parent_token is not parent_token and initiator.parent_token is not child_impl.parent_token:
                _acceptable_key_err(state, initiator, child_impl)
            check_append_token = child_impl._append_token
            check_bulk_replace_token = child_impl._bulk_replace_token if _is_collection_attribute_impl(child_impl) else None
            if initiator is not check_append_token and initiator is not check_bulk_replace_token:
                child_impl.append(child_state, child_dict, state.obj(), initiator, passive=PASSIVE_NO_FETCH)
        return child

    def emit_backref_from_collection_append_event(state, child, initiator, **kw):
        if child is None:
            return
        child_state, child_dict = (instance_state(child), instance_dict(child))
        child_impl = child_state.manager[key].impl
        if initiator.parent_token is not parent_token and initiator.parent_token is not child_impl.parent_token:
            _acceptable_key_err(state, initiator, child_impl)
        check_append_token = child_impl._append_token
        check_bulk_replace_token = child_impl._bulk_replace_token if _is_collection_attribute_impl(child_impl) else None
        if initiator is not check_append_token and initiator is not check_bulk_replace_token:
            child_impl.append(child_state, child_dict, state.obj(), initiator, passive=PASSIVE_NO_FETCH)
        return child

    def emit_backref_from_collection_remove_event(state, child, initiator, **kw):
        if child is not None and child is not PASSIVE_NO_RESULT and (child is not NO_VALUE):
            child_state, child_dict = (instance_state(child), instance_dict(child))
            child_impl = child_state.manager[key].impl
            check_replace_token: Optional[AttributeEventToken]
            if not child_impl.collection and (not child_impl.dynamic):
                check_remove_token = child_impl._remove_token
                check_replace_token = child_impl._replace_token
                check_for_dupes_on_remove = uselist and (not parent_impl.dynamic)
            else:
                check_remove_token = child_impl._remove_token
                check_replace_token = child_impl._bulk_replace_token if _is_collection_attribute_impl(child_impl) else None
                check_for_dupes_on_remove = False
            if initiator is not check_remove_token and initiator is not check_replace_token:
                if not check_for_dupes_on_remove or not util.has_dupes(state.dict[parent_impl.key], child):
                    child_impl.pop(child_state, child_dict, state.obj(), initiator, passive=PASSIVE_NO_FETCH)
    if uselist:
        event.listen(attribute, 'append', emit_backref_from_collection_append_event, retval=True, raw=True, include_key=True)
    else:
        event.listen(attribute, 'set', emit_backref_from_scalar_set_event, retval=True, raw=True, include_key=True)
    event.listen(attribute, 'remove', emit_backref_from_collection_remove_event, retval=True, raw=True, include_key=True)
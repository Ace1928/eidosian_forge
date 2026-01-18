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
def create_proxied_attribute(descriptor: Any) -> Callable[..., QueryableAttribute[Any]]:
    """Create an QueryableAttribute / user descriptor hybrid.

    Returns a new QueryableAttribute type that delegates descriptor
    behavior and getattr() to the given descriptor.
    """

    class Proxy(QueryableAttribute[Any]):
        """Presents the :class:`.QueryableAttribute` interface as a
        proxy on top of a Python descriptor / :class:`.PropComparator`
        combination.

        """
        _extra_criteria = ()

        def __init__(self, class_, key, descriptor, comparator, adapt_to_entity=None, doc=None, original_property=None):
            self.class_ = class_
            self.key = key
            self.descriptor = descriptor
            self.original_property = original_property
            self._comparator = comparator
            self._adapt_to_entity = adapt_to_entity
            self._doc = self.__doc__ = doc

        @property
        def _parententity(self):
            return inspection.inspect(self.class_, raiseerr=False)

        @property
        def parent(self):
            return inspection.inspect(self.class_, raiseerr=False)
        _is_internal_proxy = True
        _cache_key_traversal = [('key', visitors.ExtendedInternalTraversal.dp_string), ('_parententity', visitors.ExtendedInternalTraversal.dp_multi)]

        @property
        def _impl_uses_objects(self):
            return self.original_property is not None and getattr(self.class_, self.key).impl.uses_objects

        @property
        def _entity_namespace(self):
            if hasattr(self._comparator, '_parententity'):
                return self._comparator._parententity
            else:
                return AdHocHasEntityNamespace(self._parententity)

        @property
        def property(self):
            return self.comparator.property

        @util.memoized_property
        def comparator(self):
            if callable(self._comparator):
                self._comparator = self._comparator()
            if self._adapt_to_entity:
                self._comparator = self._comparator.adapt_to_entity(self._adapt_to_entity)
            return self._comparator

        def adapt_to_entity(self, adapt_to_entity):
            return self.__class__(adapt_to_entity.entity, self.key, self.descriptor, self._comparator, adapt_to_entity)

        def _clone(self, **kw):
            return self.__class__(self.class_, self.key, self.descriptor, self._comparator, adapt_to_entity=self._adapt_to_entity, original_property=self.original_property)

        def __get__(self, instance, owner):
            retval = self.descriptor.__get__(instance, owner)
            if retval is self.descriptor and instance is None:
                return self
            else:
                return retval

        def __str__(self) -> str:
            return f'{self.class_.__name__}.{self.key}'

        def __getattr__(self, attribute):
            """Delegate __getattr__ to the original descriptor and/or
            comparator."""
            try:
                return util.MemoizedSlots.__getattr__(self, attribute)
            except AttributeError:
                pass
            try:
                return getattr(descriptor, attribute)
            except AttributeError as err:
                if attribute == 'comparator':
                    raise AttributeError('comparator') from err
                try:
                    comparator = self.comparator
                except AttributeError as err2:
                    raise AttributeError('Neither %r object nor unconfigured comparator object associated with %s has an attribute %r' % (type(descriptor).__name__, self, attribute)) from err2
                else:
                    try:
                        return getattr(comparator, attribute)
                    except AttributeError as err3:
                        raise AttributeError('Neither %r object nor %r object associated with %s has an attribute %r' % (type(descriptor).__name__, type(comparator).__name__, self, attribute)) from err3
    Proxy.__name__ = type(descriptor).__name__ + 'Proxy'
    util.monkeypatch_proxied_specials(Proxy, type(descriptor), name='descriptor', from_instance=descriptor)
    return Proxy
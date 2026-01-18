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
def contains_eager(self, attr: _AttrType, alias: Optional[_FromClauseArgument]=None, _is_chain: bool=False) -> Self:
    """Indicate that the given attribute should be eagerly loaded from
        columns stated manually in the query.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        The option is used in conjunction with an explicit join that loads
        the desired rows, i.e.::

            sess.query(Order).join(Order.user).options(
                contains_eager(Order.user)
            )

        The above query would join from the ``Order`` entity to its related
        ``User`` entity, and the returned ``Order`` objects would have the
        ``Order.user`` attribute pre-populated.

        It may also be used for customizing the entries in an eagerly loaded
        collection; queries will normally want to use the
        :ref:`orm_queryguide_populate_existing` execution option assuming the
        primary collection of parent objects may already have been loaded::

            sess.query(User).join(User.addresses).filter(
                Address.email_address.like("%@aol.com")
            ).options(contains_eager(User.addresses)).populate_existing()

        See the section :ref:`contains_eager` for complete usage details.

        .. seealso::

            :ref:`loading_toplevel`

            :ref:`contains_eager`

        """
    if alias is not None:
        if not isinstance(alias, str):
            coerced_alias = coercions.expect(roles.FromClauseRole, alias)
        else:
            util.warn_deprecated("Passing a string name for the 'alias' argument to 'contains_eager()` is deprecated, and will not work in a future release.  Please use a sqlalchemy.alias() or sqlalchemy.orm.aliased() construct.", version='1.4')
            coerced_alias = alias
    elif getattr(attr, '_of_type', None):
        assert isinstance(attr, QueryableAttribute)
        ot: Optional[_InternalEntityType[Any]] = inspect(attr._of_type)
        assert ot is not None
        coerced_alias = ot.selectable
    else:
        coerced_alias = None
    cloned = self._set_relationship_strategy(attr, {'lazy': 'joined'}, propagate_to_loaders=False, opts={'eager_from_alias': coerced_alias}, _reconcile_to_other=True if _is_chain else None)
    return cloned
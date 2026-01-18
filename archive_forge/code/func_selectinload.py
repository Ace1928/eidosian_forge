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
def selectinload(self, attr: _AttrType, recursion_depth: Optional[int]=None) -> Self:
    """Indicate that the given attribute should be loaded using
        SELECT IN eager loading.

        This function is part of the :class:`_orm.Load` interface and supports
        both method-chained and standalone operation.

        examples::

            # selectin-load the "orders" collection on "User"
            select(User).options(selectinload(User.orders))

            # selectin-load Order.items and then Item.keywords
            select(Order).options(
                selectinload(Order.items).selectinload(Item.keywords)
            )

            # lazily load Order.items, but when Items are loaded,
            # selectin-load the keywords collection
            select(Order).options(
                lazyload(Order.items).selectinload(Item.keywords)
            )

        :param recursion_depth: optional int; when set to a positive integer
         in conjunction with a self-referential relationship,
         indicates "selectin" loading will continue that many levels deep
         automatically until no items are found.

         .. note:: The :paramref:`_orm.selectinload.recursion_depth` option
            currently supports only self-referential relationships.  There
            is not yet an option to automatically traverse recursive structures
            with more than one relationship involved.

            Additionally, the :paramref:`_orm.selectinload.recursion_depth`
            parameter is new and experimental and should be treated as "alpha"
            status for the 2.0 series.

         .. versionadded:: 2.0 added
            :paramref:`_orm.selectinload.recursion_depth`


        .. seealso::

            :ref:`loading_toplevel`

            :ref:`selectin_eager_loading`

        """
    return self._set_relationship_strategy(attr, {'lazy': 'selectin'}, opts={'recursion_depth': recursion_depth})
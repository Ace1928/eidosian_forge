from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
def attribute_keyed_dict(attr_name: str, *, ignore_unpopulated_attribute: bool=False) -> Type[KeyFuncDict[Any, Any]]:
    """A dictionary-based collection type with attribute-based keying.

    .. versionchanged:: 2.0 Renamed :data:`.attribute_mapped_collection` to
       :func:`.attribute_keyed_dict`.

    Returns a :class:`.KeyFuncDict` factory which will produce new
    dictionary keys based on the value of a particular named attribute on
    ORM mapped instances to be added to the dictionary.

    .. note:: the value of the target attribute must be assigned with its
       value at the time that the object is being added to the
       dictionary collection.   Additionally, changes to the key attribute
       are **not tracked**, which means the key in the dictionary is not
       automatically synchronized with the key value on the target object
       itself.  See :ref:`key_collections_mutations` for further details.

    .. seealso::

        :ref:`orm_dictionary_collection` - background on use

    :param attr_name: string name of an ORM-mapped attribute
     on the mapped class, the value of which on a particular instance
     is to be used as the key for a new dictionary entry for that instance.
    :param ignore_unpopulated_attribute:  if True, and the target attribute
     on an object is not populated at all, the operation will be silently
     skipped.  By default, an error is raised.

     .. versionadded:: 2.0 an error is raised by default if the attribute
        being used for the dictionary key is determined that it was never
        populated with any value.  The
        :paramref:`_orm.attribute_keyed_dict.ignore_unpopulated_attribute`
        parameter may be set which will instead indicate that this condition
        should be ignored, and the append operation silently skipped.
        This is in contrast to the behavior of the 1.x series which would
        erroneously populate the value in the dictionary with an arbitrary key
        value of ``None``.


    """
    return _mapped_collection_cls(_AttrGetter(attr_name), ignore_unpopulated_attribute=ignore_unpopulated_attribute)
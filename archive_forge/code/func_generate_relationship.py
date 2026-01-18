from ``Engineer`` to ``Employee``, we need to set up both the relationship
from __future__ import annotations
import dataclasses
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import util
from ..orm import backref
from ..orm import declarative_base as _declarative_base
from ..orm import exc as orm_exc
from ..orm import interfaces
from ..orm import relationship
from ..orm.decl_base import _DeferredMapperConfig
from ..orm.mapper import _CONFIGURE_MUTEX
from ..schema import ForeignKeyConstraint
from ..sql import and_
from ..util import Properties
from ..util.typing import Protocol
def generate_relationship(base: Type[Any], direction: RelationshipDirection, return_fn: Union[Callable[..., Relationship[Any]], Callable[..., ORMBackrefArgument]], attrname: str, local_cls: Type[Any], referred_cls: Type[Any], **kw: Any) -> Union[Relationship[Any], ORMBackrefArgument]:
    """Generate a :func:`_orm.relationship` or :func:`.backref`
    on behalf of two
    mapped classes.

    An alternate implementation of this function can be specified using the
    :paramref:`.AutomapBase.prepare.generate_relationship` parameter.

    The default implementation of this function is as follows::

        if return_fn is backref:
            return return_fn(attrname, **kw)
        elif return_fn is relationship:
            return return_fn(referred_cls, **kw)
        else:
            raise TypeError("Unknown relationship function: %s" % return_fn)

    :param base: the :class:`.AutomapBase` class doing the prepare.

    :param direction: indicate the "direction" of the relationship; this will
     be one of :data:`.ONETOMANY`, :data:`.MANYTOONE`, :data:`.MANYTOMANY`.

    :param return_fn: the function that is used by default to create the
     relationship.  This will be either :func:`_orm.relationship` or
     :func:`.backref`.  The :func:`.backref` function's result will be used to
     produce a new :func:`_orm.relationship` in a second step,
     so it is critical
     that user-defined implementations correctly differentiate between the two
     functions, if a custom relationship function is being used.

    :param attrname: the attribute name to which this relationship is being
     assigned. If the value of :paramref:`.generate_relationship.return_fn` is
     the :func:`.backref` function, then this name is the name that is being
     assigned to the backref.

    :param local_cls: the "local" class to which this relationship or backref
     will be locally present.

    :param referred_cls: the "referred" class to which the relationship or
     backref refers to.

    :param \\**kw: all additional keyword arguments are passed along to the
     function.

    :return: a :func:`_orm.relationship` or :func:`.backref` construct,
     as dictated
     by the :paramref:`.generate_relationship.return_fn` parameter.

    """
    if return_fn is backref:
        return return_fn(attrname, **kw)
    elif return_fn is relationship:
        return return_fn(referred_cls, **kw)
    else:
        raise TypeError('Unknown relationship function: %s' % return_fn)
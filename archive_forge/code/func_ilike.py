from __future__ import annotations
from enum import IntEnum
from operator import add as _uncast_add
from operator import and_ as _uncast_and_
from operator import contains as _uncast_contains
from operator import eq as _uncast_eq
from operator import floordiv as _uncast_floordiv
from operator import ge as _uncast_ge
from operator import getitem as _uncast_getitem
from operator import gt as _uncast_gt
from operator import inv as _uncast_inv
from operator import le as _uncast_le
from operator import lshift as _uncast_lshift
from operator import lt as _uncast_lt
from operator import mod as _uncast_mod
from operator import mul as _uncast_mul
from operator import ne as _uncast_ne
from operator import neg as _uncast_neg
from operator import or_ as _uncast_or_
from operator import rshift as _uncast_rshift
from operator import sub as _uncast_sub
from operator import truediv as _uncast_truediv
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def ilike(self, other: Any, escape: Optional[str]=None) -> ColumnOperators:
    """Implement the ``ilike`` operator, e.g. case insensitive LIKE.

        In a column context, produces an expression either of the form::

            lower(a) LIKE lower(other)

        Or on backends that support the ILIKE operator::

            a ILIKE other

        E.g.::

            stmt = select(sometable).\\
                where(sometable.c.column.ilike("%foobar%"))

        :param other: expression to be compared
        :param escape: optional escape character, renders the ``ESCAPE``
          keyword, e.g.::

            somecolumn.ilike("foo/%bar", escape="/")

        .. seealso::

            :meth:`.ColumnOperators.like`

        """
    return self.operate(ilike_op, other, escape=escape)
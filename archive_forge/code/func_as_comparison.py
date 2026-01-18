from __future__ import annotations
import datetime
import decimal
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import annotation
from . import coercions
from . import operators
from . import roles
from . import schema
from . import sqltypes
from . import type_api
from . import util as sqlutil
from ._typing import is_table_value_type
from .base import _entity_namespace
from .base import ColumnCollection
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .elements import _type_from_args
from .elements import BinaryExpression
from .elements import BindParameter
from .elements import Cast
from .elements import ClauseList
from .elements import ColumnElement
from .elements import Extract
from .elements import FunctionFilter
from .elements import Grouping
from .elements import literal_column
from .elements import NamedColumn
from .elements import Over
from .elements import WithinGroup
from .selectable import FromClause
from .selectable import Select
from .selectable import TableValuedAlias
from .sqltypes import TableValueType
from .type_api import TypeEngine
from .visitors import InternalTraversal
from .. import util
def as_comparison(self, left_index: int, right_index: int) -> FunctionAsBinary:
    """Interpret this expression as a boolean comparison between two
        values.

        This method is used for an ORM use case described at
        :ref:`relationship_custom_operator_sql_function`.

        A hypothetical SQL function "is_equal()" which compares to values
        for equality would be written in the Core expression language as::

            expr = func.is_equal("a", "b")

        If "is_equal()" above is comparing "a" and "b" for equality, the
        :meth:`.FunctionElement.as_comparison` method would be invoked as::

            expr = func.is_equal("a", "b").as_comparison(1, 2)

        Where above, the integer value "1" refers to the first argument of the
        "is_equal()" function and the integer value "2" refers to the second.

        This would create a :class:`.BinaryExpression` that is equivalent to::

            BinaryExpression("a", "b", operator=op.eq)

        However, at the SQL level it would still render as
        "is_equal('a', 'b')".

        The ORM, when it loads a related object or collection, needs to be able
        to manipulate the "left" and "right" sides of the ON clause of a JOIN
        expression. The purpose of this method is to provide a SQL function
        construct that can also supply this information to the ORM, when used
        with the :paramref:`_orm.relationship.primaryjoin` parameter. The
        return value is a containment object called :class:`.FunctionAsBinary`.

        An ORM example is as follows::

            class Venue(Base):
                __tablename__ = 'venue'
                id = Column(Integer, primary_key=True)
                name = Column(String)

                descendants = relationship(
                    "Venue",
                    primaryjoin=func.instr(
                        remote(foreign(name)), name + "/"
                    ).as_comparison(1, 2) == 1,
                    viewonly=True,
                    order_by=name
                )

        Above, the "Venue" class can load descendant "Venue" objects by
        determining if the name of the parent Venue is contained within the
        start of the hypothetical descendant value's name, e.g. "parent1" would
        match up to "parent1/child1", but not to "parent2/child1".

        Possible use cases include the "materialized path" example given above,
        as well as making use of special SQL functions such as geometric
        functions to create join conditions.

        :param left_index: the integer 1-based index of the function argument
         that serves as the "left" side of the expression.
        :param right_index: the integer 1-based index of the function argument
         that serves as the "right" side of the expression.

        .. versionadded:: 1.3

        .. seealso::

            :ref:`relationship_custom_operator_sql_function` -
            example use within the ORM

        """
    return FunctionAsBinary(self, left_index, right_index)
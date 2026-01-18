from enum import Enum
from typing import (
from ..error import GraphQLError
from ..language import (
from ..pyutils import (
from ..utilities.value_from_ast_untyped import value_from_ast_untyped
from .assert_name import assert_name, assert_enum_value_name
@staticmethod
def out_type(value: Dict[str, Any]) -> Any:
    """Transform outbound values (this is an extension of GraphQL.js).

        This default implementation passes values unaltered as dictionaries.
        """
    return value
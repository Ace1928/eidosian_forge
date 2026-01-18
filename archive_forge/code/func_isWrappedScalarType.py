from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def isWrappedScalarType(typ: Type) -> bool:
    """
    Given a type, determine if it is a c10::scalar which we will wrap in a lazy Value.
    Since we literally change the type from scalarT to valueT, information is lost.
    This function helps build a list of wrapped scalars to save that information
    """
    if isinstance(typ, BaseType):
        return typ.name == BaseTy.Scalar
    elif isinstance(typ, (OptionalType, ListType)):
        return isWrappedScalarType(typ.elem)
    return False
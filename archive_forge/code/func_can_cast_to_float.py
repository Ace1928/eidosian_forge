from typing import Tuple, Union
from langchain.chains.query_constructor.ir import (
def can_cast_to_float(string: str) -> bool:
    """Check if a string can be cast to a float."""
    try:
        float(string)
        return True
    except ValueError:
        return False
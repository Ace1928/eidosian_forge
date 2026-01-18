from __future__ import annotations
import keyword
import warnings
from typing import Collection, List, Mapping, Optional, Set, Tuple, Union
@staticmethod
def check_axis_name(name: str) -> bool:
    """Check if the name is a valid axis name.

        Args:
            name (str): the axis name to check

        Returns:
            bool: whether the axis name is valid
        """
    is_valid, _ = ParsedExpression.check_axis_name_return_reason(name)
    return is_valid
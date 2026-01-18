from __future__ import annotations
import keyword
import warnings
from typing import Collection, List, Mapping, Optional, Set, Tuple, Union
@staticmethod
def check_axis_name_return_reason(name: str, allow_underscore: bool=False) -> Tuple[bool, str]:
    """Check if the given axis name is valid, and a message explaining why if not.

        Valid axes names are python identifiers except keywords, and should not start or end with an underscore.

        Args:
            name (str): the axis name to check
            allow_underscore (bool): whether axis names are allowed to start with an underscore

        Returns:
            Tuple[bool, str]: whether the axis name is valid, a message explaining why if not
        """
    if not str.isidentifier(name):
        return (False, 'not a valid python identifier')
    elif name[0] == '_' or name[-1] == '_':
        if name == '_' and allow_underscore:
            return (True, '')
        return (False, 'axis name should should not start or end with underscore')
    else:
        if keyword.iskeyword(name):
            warnings.warn(f'It is discouraged to use axes names that are keywords: {name}', RuntimeWarning)
        if name in ['axis']:
            warnings.warn("It is discouraged to use 'axis' as an axis name and will raise an error in future", FutureWarning)
        return (True, '')
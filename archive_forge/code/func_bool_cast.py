from typing import Union
import click
import functools
def bool_cast(string: str) -> Union[bool, str]:
    """Cast a string to a boolean if possible, otherwise return the string."""
    if string.lower() == 'true' or string == '1':
        return True
    elif string.lower() == 'false' or string == '0':
        return False
    else:
        return string
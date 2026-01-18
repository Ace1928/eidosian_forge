import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def get_old_value(self, args: Sequence[Any], kwargs: Dict[str, Any], default: Any=None) -> Any:
    """Returns the old value of the named argument without replacing it.

        Returns ``default`` if the argument is not present.
        """
    if self.arg_pos is not None and len(args) > self.arg_pos:
        return args[self.arg_pos]
    else:
        return kwargs.get(self.name, default)
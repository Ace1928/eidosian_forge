import lz4
import io
import os
import builtins
import sys
from ._frame import (  # noqa: F401
def has_context(self):
    """Return whether the compression context exists.

        Returns:
            bool: ``True`` if the compression context exists, ``False``
                otherwise.
        """
    return self._context is not None
import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def isLinux(self) -> bool:
    """
        Check if current platform is Linux.

        @return: C{True} if the current platform has been detected as Linux.
        """
    return self._platform.startswith('linux')
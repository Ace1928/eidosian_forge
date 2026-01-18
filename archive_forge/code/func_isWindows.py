import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def isWindows(self) -> bool:
    """
        Are we running in Windows?

        @return: C{True} if the current platform has been detected as
            Windows.
        """
    return self.getType() == 'win32'
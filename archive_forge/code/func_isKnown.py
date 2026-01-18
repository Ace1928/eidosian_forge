import os
import sys
import warnings
from time import time as seconds
from typing import Optional
def isKnown(self) -> bool:
    """
        Do we know about this platform?

        @return: Boolean indicating whether this is a known platform or not.
        """
    return self.type != None
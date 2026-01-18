import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
def readLines(self, lines=10):
    """Read a list of lines from the log file.

        This doesn't returns all of the files lines - call it multiple times.
        """
    result = []
    for i in range(lines):
        line = self._file.readline()
        if not line:
            break
        result.append(line)
    return result
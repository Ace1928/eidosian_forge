import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
@classmethod
def fromFullPath(cls, filename, *args, **kwargs):
    """
        Construct a log file from a full file path.
        """
    logPath = os.path.abspath(filename)
    return cls(os.path.basename(logPath), os.path.dirname(logPath), *args, **kwargs)
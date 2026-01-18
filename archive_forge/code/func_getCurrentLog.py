import glob
import os
import stat
import time
from typing import BinaryIO, Optional, cast
from twisted.python import threadable
def getCurrentLog(self):
    """
        Return a LogReader for the current log file.
        """
    return LogReader(self.path)
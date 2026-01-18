from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def islnk(self):
    """Return True if it is a hard link."""
    return self.type == LNKTYPE
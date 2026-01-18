from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def get_rootentry_name(self):
    """
        Return root entry name. Should usually be 'Root Entry' or 'R' in most
        implementations.
        """
    return self.root.name
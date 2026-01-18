import sys
import traceback
import time
from io import StringIO
import linecache
from paste.exceptions import serial_number_generator
import warnings
def getRevision(self, globals):
    if not self.show_revisions:
        return None
    revision = globals.get('__revision__', None)
    if revision is None:
        revision = globals.get('__version__', None)
    if revision is not None:
        try:
            revision = str(revision).strip()
        except:
            revision = '???'
    return revision
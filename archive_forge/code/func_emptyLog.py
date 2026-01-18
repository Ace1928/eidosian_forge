import sys
import time
from uuid import UUID
import pytest
from cherrypy._cpcompat import text_or_bytes
def emptyLog(self):
    """Overwrite self.logfile with 0 bytes."""
    with open(self.logfile, 'wb') as f:
        f.write('')
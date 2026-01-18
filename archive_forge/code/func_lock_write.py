import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def lock_write(self, relpath):
    """Lock the given file for exclusive access.
        :return: A lock object, which should be passed to Transport.unlock()
        """
    return self.lock_read(relpath)
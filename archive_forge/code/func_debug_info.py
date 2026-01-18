import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def debug_info(self, c):
    """
        Return some info --- useful to spot problems with refcounting
        """
    with self.mutex:
        result = []
        keys = list(self.id_to_refcount.keys())
        keys.sort()
        for ident in keys:
            if ident != '0':
                result.append('  %s:       refcount=%s\n    %s' % (ident, self.id_to_refcount[ident], str(self.id_to_obj[ident][0])[:75]))
        return '\n'.join(result)
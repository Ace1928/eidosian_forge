import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def ext_convert(self, value):
    """Default converter for the ext:// protocol."""
    return self.resolve(value)
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
def configure_custom(self, config):
    """Configure an object with a user-supplied factory."""
    c = config.pop('()')
    if not callable(c):
        c = self.resolve(c)
    kwargs = {k: config[k] for k in config if k != '.' and valid_ident(k)}
    result = c(**kwargs)
    props = config.pop('.', None)
    if props:
        for name, value in props.items():
            setattr(result, name, value)
    return result
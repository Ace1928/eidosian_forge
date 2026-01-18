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
def cfg_convert(self, value):
    """Default converter for the cfg:// protocol."""
    rest = value
    m = self.WORD_PATTERN.match(rest)
    if m is None:
        raise ValueError('Unable to convert %r' % value)
    else:
        rest = rest[m.end():]
        d = self.config[m.groups()[0]]
        while rest:
            m = self.DOT_PATTERN.match(rest)
            if m:
                d = d[m.groups()[0]]
            else:
                m = self.INDEX_PATTERN.match(rest)
                if m:
                    idx = m.groups()[0]
                    if not self.DIGIT_PATTERN.match(idx):
                        d = d[idx]
                    else:
                        try:
                            n = int(idx)
                            d = d[n]
                        except TypeError:
                            d = d[idx]
            if m:
                rest = rest[m.end():]
            else:
                raise ValueError('Unable to convert %r at %r' % (value, rest))
    return d
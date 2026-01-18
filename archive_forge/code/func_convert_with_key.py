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
def convert_with_key(self, key, value, replace=True):
    result = self.configurator.convert(value)
    if value is not result:
        if replace:
            self[key] = result
        if type(result) in (ConvertingDict, ConvertingList, ConvertingTuple):
            result.parent = self
            result.key = key
    return result
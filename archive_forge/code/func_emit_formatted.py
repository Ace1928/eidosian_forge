from __future__ import (absolute_import, division, print_function)
import os
import socket
import random
import time
import uuid
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
def emit_formatted(self, record):
    if self.flatten:
        results = flatdict.FlatDict(record)
        self.emit(self._dump_results(results))
    else:
        self.emit(self._dump_results(record))
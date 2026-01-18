import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def emit_pid(self, name, pid):
    """Adds a process metadata event to the trace.

    Args:
      name:  The process name as a string.
      pid:  Identifier of the process as an integer.
    """
    event = {}
    event['name'] = 'process_name'
    event['ph'] = 'M'
    event['pid'] = pid
    event['args'] = {'name': name}
    self._metadata.append(event)
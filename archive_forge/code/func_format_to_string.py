import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def format_to_string(self, pretty=False):
    """Formats the chrome trace to a string.

    Args:
      pretty: (Optional.)  If True, produce human-readable JSON output.

    Returns:
      A JSON-formatted string in Chrome Trace format.
    """
    trace = {}
    trace['traceEvents'] = self._metadata + self._events
    if pretty:
        return json.dumps(trace, indent=4, separators=(',', ': '))
    else:
        return json.dumps(trace, separators=(',', ':'))
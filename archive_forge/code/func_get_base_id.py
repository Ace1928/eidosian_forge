import collections
import datetime
import functools
import inspect
import socket
import threading
from oslo_utils import reflection
from oslo_utils import uuidutils
from osprofiler import _utils as utils
from osprofiler import notifier
def get_base_id(self):
    """Return base id of a trace.

        Base id is the same for all elements in one trace. It's main goal is
        to be able to retrieve by one request all trace elements from storage.
        """
    return self._trace_stack[0]
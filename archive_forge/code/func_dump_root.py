import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
@property
def dump_root(self):
    return self._dump_root
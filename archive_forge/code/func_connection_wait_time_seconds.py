from __future__ import annotations
from queue import Empty
from kombu.transport import virtual
from kombu.utils import cached_property
from kombu.utils.encoding import str_to_bytes
from kombu.utils.json import dumps, loads
from kombu.log import get_logger
@cached_property
def connection_wait_time_seconds(self):
    return self.options.get('connection_wait_time_seconds', self.default_connection_wait_time_seconds)
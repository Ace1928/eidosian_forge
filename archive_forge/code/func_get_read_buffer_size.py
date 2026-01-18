import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def get_read_buffer_size(self):
    """Return the current size of the read buffer."""
    return self._ssl_protocol._get_read_buffer_size()
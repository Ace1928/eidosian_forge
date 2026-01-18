import logging
import threading
import numpy as np
from ..core import Format, image_as_uint
from ..core.request import URI_FILE, URI_BYTES
from .pillowmulti import GIFFormat, TIFFFormat  # noqa: E402, F401
@property
def plugin_id(self):
    """The PIL plugin id."""
    return self._plugin_id
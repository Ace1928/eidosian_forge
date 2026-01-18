from __future__ import annotations
import re
from typing import Match, cast
from zmq.backend import zmq_version_info
def pyzmq_version() -> str:
    """return the version of pyzmq as a string"""
    if __revision__:
        return '+'.join([__version__, __revision__[:6]])
    else:
        return __version__
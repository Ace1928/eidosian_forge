import subprocess
from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger
@property
def returncode(self):
    return self._transport.get_returncode()
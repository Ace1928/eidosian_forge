import subprocess
from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger
def process_exited(self):
    self._process_exited = True
    self._maybe_close_transport()
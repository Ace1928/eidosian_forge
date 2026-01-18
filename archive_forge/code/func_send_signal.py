import subprocess
from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger
def send_signal(self, signal):
    self._transport.send_signal(signal)
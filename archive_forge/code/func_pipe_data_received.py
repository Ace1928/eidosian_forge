import subprocess
from . import events
from . import protocols
from . import streams
from . import tasks
from .log import logger
def pipe_data_received(self, fd, data):
    if fd == 1:
        reader = self.stdout
    elif fd == 2:
        reader = self.stderr
    else:
        reader = None
    if reader is not None:
        reader.feed_data(data)
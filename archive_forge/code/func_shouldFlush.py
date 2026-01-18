import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def shouldFlush(self, record):
    """
        Check for buffer full or a record at the flushLevel or higher.
        """
    return len(self.buffer) >= self.capacity or record.levelno >= self.flushLevel
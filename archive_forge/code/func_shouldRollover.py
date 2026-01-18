import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def shouldRollover(self, record):
    """
        Determine if rollover should occur.

        record is not used, as we are just comparing times, but it is needed so
        the method signatures are the same
        """
    t = int(time.time())
    if t >= self.rolloverAt:
        if os.path.exists(self.baseFilename) and (not os.path.isfile(self.baseFilename)):
            self.rolloverAt = self.computeRollover(t)
            return False
        return True
    return False
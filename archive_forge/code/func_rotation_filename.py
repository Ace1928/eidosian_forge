import io, logging, socket, os, pickle, struct, time, re
from stat import ST_DEV, ST_INO, ST_MTIME
import queue
import threading
import copy
def rotation_filename(self, default_name):
    """
        Modify the filename of a log file when rotating.

        This is provided so that a custom filename can be provided.

        The default implementation calls the 'namer' attribute of the
        handler, if it's callable, passing the default name to
        it. If the attribute isn't callable (the default is None), the name
        is returned unchanged.

        :param default_name: The default name for the log file.
        """
    if not callable(self.namer):
        result = default_name
    else:
        result = self.namer(default_name)
    return result
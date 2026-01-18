import inspect
import os
import sys
def prepend_to(self, key, entry):
    """Prepends a new entry to a PATH-style environment variable, creating
        it if it doesn't exist already.
        """
    try:
        tail = os.path.pathsep + self[key]
    except KeyError:
        tail = ''
    self[key] = entry + tail
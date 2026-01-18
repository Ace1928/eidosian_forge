import functools
import re
import sys
from Xlib.support import lock
def insert_resources(self, resources):
    """insert_resources(resources)

        Insert all resources entries in the list RESOURCES into the
        database.  Each element in RESOURCES should be a tuple:

          (resource, value)

        Where RESOURCE is a string and VALUE can be any Python value.

        """
    for res, value in resources:
        self.insert(res, value)
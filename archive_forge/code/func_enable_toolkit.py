import sys
import os
from os import path
from contextlib import contextmanager
@enable_toolkit.setter
def enable_toolkit(self, toolkit):
    """
        Deprecated.

        Property setter for the Enable toolkit.  The toolkit can be set more
        than once, but only if it is the same one each time.  An application
        that is written for a particular toolkit can explicitly set it before
        any other module that gets the value is imported.
        """
    from warnings import warn
    warn('Use of the enable_toolkit attribute is deprecated.')
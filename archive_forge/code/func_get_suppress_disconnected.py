import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def get_suppress_disconnected(self, val):
    """Get if suppress disconnected is set.

        Refer to set_suppress_disconnected for more information.
        """
    return self.obj_dict['suppress_disconnected']
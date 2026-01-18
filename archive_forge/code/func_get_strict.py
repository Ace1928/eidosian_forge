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
def get_strict(self, val):
    """Get graph's 'strict' mode (True, False).

        This option is only valid for top level graphs.
        """
    return self.obj_dict['strict']
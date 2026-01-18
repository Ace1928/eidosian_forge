import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def cpu_dispatch_names(self):
    """
        return a list of final CPU dispatch feature names
        """
    return self.parse_dispatch_names
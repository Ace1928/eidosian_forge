import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def cpu_baseline_names(self):
    """
        return a list of final CPU baseline feature names
        """
    return self.parse_baseline_names
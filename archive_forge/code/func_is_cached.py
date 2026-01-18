import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def is_cached(self):
    """
        Returns True if the class loaded from the cache file
        """
    return self.cache_infile and self.hit_cache
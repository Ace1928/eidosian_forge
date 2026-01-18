import fnmatch
import logging
import os
import re
import sys
from . import DistlibException
from .compat import fsdecode
from .util import convert_path
def add_many(self, items):
    """
        Add a list of files to the manifest.

        :param items: The pathnames to add. These can be relative to the base.
        """
    for item in items:
        self.add(item)
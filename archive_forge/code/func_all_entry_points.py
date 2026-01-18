import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def all_entry_points(self):
    if not hasattr(self, '_entry_points'):
        self._entry_points = list(pkg_resources.iter_entry_points('paste.paster_create_template'))
    return self._entry_points
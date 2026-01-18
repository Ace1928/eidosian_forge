import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
def get_group_description(self, group):
    for entry in pkg_resources.iter_entry_points('paste.entry_point_description'):
        if entry.name == group:
            ep = entry.load()
            if hasattr(ep, 'description'):
                return ep.description
            else:
                return ep
    return None
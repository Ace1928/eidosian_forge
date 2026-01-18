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
def get_groups_by_pattern(self, pattern):
    env = pkg_resources.Environment()
    eps = {}
    for project_name in env:
        for dist in env[project_name]:
            for name in pkg_resources.get_entry_map(dist):
                if pattern and (not pattern.search(name)):
                    continue
                if not pattern and name.startswith('paste.description.'):
                    continue
                eps[name] = None
    return sorted(eps.keys())
import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def inspect_files(self, output_dir, templates, vars):
    file_sources = {}
    for template in templates:
        self._find_files(template, vars, file_sources)
    self._show_files(output_dir, file_sources)
    self._show_leftovers(output_dir, file_sources)
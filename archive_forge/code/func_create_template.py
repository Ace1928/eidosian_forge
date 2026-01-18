import fnmatch
import os
import re
import sys
import pkg_resources
from . import copydir
from . import pluginlib
from .command import Command, BadCommand
def create_template(self, template, output_dir, vars):
    if self.verbose:
        print('Creating template %s' % template.name)
    template.run(self, output_dir, vars)
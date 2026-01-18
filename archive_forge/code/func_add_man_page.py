import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
def add_man_page(self, man_page):
    self.data_files = "%s\n  '%s'" % (self.data_files, man_page)
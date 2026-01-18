import sys
import os
import re
import pkg_resources
from string import Template
def normalize_output_dir(self, dest):
    return os.path.abspath(os.path.normpath(dest))
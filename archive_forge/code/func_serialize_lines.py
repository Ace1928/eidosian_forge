from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def serialize_lines(self):
    return self.prefix_lines + ["    {0} = r'''".format(self.name)] + self.lines + ["'''"]
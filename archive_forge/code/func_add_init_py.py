from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def add_init_py(path):
    path = os.path.join(path, '__init__.py')
    if os.path.exists(path):
        return
    with open(path, 'wb') as f:
        f.write(b'')
    files_to_remove.append(path)
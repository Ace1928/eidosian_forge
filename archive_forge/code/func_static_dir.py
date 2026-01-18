import json
import os.path as osp
from itertools import filterfalse
from .jlpmapp import HERE
@static_dir.setter
def static_dir(self, static_dir):
    self._data['jupyterlab']['staticDir'] = static_dir
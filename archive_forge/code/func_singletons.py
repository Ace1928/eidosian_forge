import json
import os.path as osp
from itertools import filterfalse
from .jlpmapp import HERE
@property
def singletons(self):
    """A dict mapping all singleton names to their semver"""
    data = self._data
    return {k: data['resolutions'].get(k, None) for k in data['jupyterlab']['singletonPackages']}
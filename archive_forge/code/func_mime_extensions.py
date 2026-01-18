import json
import os.path as osp
from itertools import filterfalse
from .jlpmapp import HERE
@property
def mime_extensions(self):
    """A dict mapping all MIME extension names to their semver"""
    data = self._data
    return {k: data['resolutions'][k] for k in data['jupyterlab']['mimeExtensions']}
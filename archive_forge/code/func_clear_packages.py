import json
import os.path as osp
from itertools import filterfalse
from .jlpmapp import HERE
def clear_packages(self, lab_only=True):
    """Clear the packages/extensions."""
    data = self._data
    if lab_only:
        data['dependencies'] = _only_nonlab(data['dependencies'])
        data['resolutions'] = _only_nonlab(data['resolutions'])
        data['jupyterlab']['extensions'] = _only_nonlab(data['jupyterlab']['extensions'])
        data['jupyterlab']['mimeExtensions'] = _only_nonlab(data['jupyterlab']['mimeExtensions'])
        data['jupyterlab']['singletonPackages'] = _only_nonlab(data['jupyterlab']['singletonPackages'])
    else:
        data['dependencies'] = {}
        data['resolutions'] = {}
        data['jupyterlab']['extensions'] = {}
        data['jupyterlab']['mimeExtensions'] = {}
        data['jupyterlab']['singletonPackages'] = []
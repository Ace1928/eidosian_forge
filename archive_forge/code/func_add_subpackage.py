import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def add_subpackage(self, subpackage_name, subpackage_path=None, standalone=False):
    """Add a sub-package to the current Configuration instance.

        This is useful in a setup.py script for adding sub-packages to a
        package.

        Parameters
        ----------
        subpackage_name : str
            name of the subpackage
        subpackage_path : str
            if given, the subpackage path such as the subpackage is in
            subpackage_path / subpackage_name. If None,the subpackage is
            assumed to be located in the local path / subpackage_name.
        standalone : bool
        """
    if standalone:
        parent_name = None
    else:
        parent_name = self.name
    config_list = self.get_subpackage(subpackage_name, subpackage_path, parent_name=parent_name, caller_level=2)
    if not config_list:
        self.warn('No configuration returned, assuming unavailable.')
    for config in config_list:
        d = config
        if isinstance(config, Configuration):
            d = config.todict()
        assert isinstance(d, dict), repr(type(d))
        self.info('Appending %s configuration to %s' % (d.get('name'), self.name))
        self.dict_append(**d)
    dist = self.get_distribution()
    if dist is not None:
        self.warn('distutils distribution has been initialized, it may be too late to add a subpackage ' + subpackage_name)
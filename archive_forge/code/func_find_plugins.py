import sys
import os
import io
import time
import re
import types
from typing import Protocol
import zipfile
import zipimport
import warnings
import stat
import functools
import pkgutil
import operator
import platform
import collections
import plistlib
import email.parser
import errno
import tempfile
import textwrap
import inspect
import ntpath
import posixpath
import importlib
import importlib.machinery
from pkgutil import get_importer
import _imp
from os import utime
from os import open as os_open
from os.path import isdir, split
from pkg_resources.extern.jaraco.text import (
from pkg_resources.extern import platformdirs
from pkg_resources.extern import packaging
def find_plugins(self, plugin_env, full_env=None, installer=None, fallback=True):
    """Find all activatable distributions in `plugin_env`

        Example usage::

            distributions, errors = working_set.find_plugins(
                Environment(plugin_dirlist)
            )
            # add plugins+libs to sys.path
            map(working_set.add, distributions)
            # display errors
            print('Could not load', errors)

        The `plugin_env` should be an ``Environment`` instance that contains
        only distributions that are in the project's "plugin directory" or
        directories. The `full_env`, if supplied, should be an ``Environment``
        contains all currently-available distributions.  If `full_env` is not
        supplied, one is created automatically from the ``WorkingSet`` this
        method is called on, which will typically mean that every directory on
        ``sys.path`` will be scanned for distributions.

        `installer` is a standard installer callback as used by the
        ``resolve()`` method. The `fallback` flag indicates whether we should
        attempt to resolve older versions of a plugin if the newest version
        cannot be resolved.

        This method returns a 2-tuple: (`distributions`, `error_info`), where
        `distributions` is a list of the distributions found in `plugin_env`
        that were loadable, along with any other distributions that are needed
        to resolve their dependencies.  `error_info` is a dictionary mapping
        unloadable plugin distributions to an exception instance describing the
        error that occurred. Usually this will be a ``DistributionNotFound`` or
        ``VersionConflict`` instance.
        """
    plugin_projects = list(plugin_env)
    plugin_projects.sort()
    error_info = {}
    distributions = {}
    if full_env is None:
        env = Environment(self.entries)
        env += plugin_env
    else:
        env = full_env + plugin_env
    shadow_set = self.__class__([])
    list(map(shadow_set.add, self))
    for project_name in plugin_projects:
        for dist in plugin_env[project_name]:
            req = [dist.as_requirement()]
            try:
                resolvees = shadow_set.resolve(req, env, installer)
            except ResolutionError as v:
                error_info[dist] = v
                if fallback:
                    continue
                else:
                    break
            else:
                list(map(shadow_set.add, resolvees))
                distributions.update(dict.fromkeys(resolvees))
                break
    distributions = list(distributions)
    distributions.sort()
    return (distributions, error_info)
import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
@__clear_cache
def register_opt(self, opt, group=None, cli=False):
    """Register an option schema.

        Registering an option schema makes any option value which is previously
        or subsequently parsed from the command line or config files available
        as an attribute of this object.

        :param opt: an instance of an Opt sub-class
        :param group: an optional OptGroup object or group name
        :param cli: whether this is a CLI option
        :return: False if the opt was already registered, True otherwise
        :raises: DuplicateOptError
        """
    if group is not None:
        group = self._get_group(group, autocreate=True)
        if cli:
            self._add_cli_opt(opt, group)
        self._track_deprecated_opts(opt, group=group)
        return group._register_opt(opt, cli)
    if group is None:
        if opt.name in self.disallow_names:
            raise ValueError('Name %s was reserved for oslo.config.' % opt.name)
    if cli:
        self._add_cli_opt(opt, None)
    if _is_opt_registered(self._opts, opt):
        return False
    self._opts[opt.dest] = {'opt': opt, 'cli': cli}
    self._track_deprecated_opts(opt)
    return True
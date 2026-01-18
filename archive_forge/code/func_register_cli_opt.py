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
def register_cli_opt(self, opt, group=None):
    """Register a CLI option schema.

        CLI option schemas must be registered before the command line and
        config files are parsed. This is to ensure that all CLI options are
        shown in --help and option validation works as expected.

        :param opt: an instance of an Opt sub-class
        :param group: an optional OptGroup object or group name
        :return: False if the opt was already registered, True otherwise
        :raises: DuplicateOptError, ArgsAlreadyParsedError
        """
    if self._args is not None:
        raise ArgsAlreadyParsedError('cannot register CLI option')
    return self.register_opt(opt, group, cli=True, clear_cache=False)
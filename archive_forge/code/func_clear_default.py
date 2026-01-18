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
def clear_default(self, name, group=None):
    """Clear an override an opt's default value.

        Clear a previously set override of the default value of given option.

        :param name: the name/dest of the opt
        :param group: an option OptGroup object or group name
        :raises: NoSuchOptError, NoSuchGroupError
        """
    opt_info = self._get_opt_info(name, group)
    opt_info.pop('default', None)
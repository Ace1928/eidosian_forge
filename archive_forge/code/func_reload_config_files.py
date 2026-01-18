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
@__clear_drivers_cache
def reload_config_files(self):
    """Reload configure files and parse all options

        :return: False if reload configure files failed or else return True
        """
    try:
        namespace = self._reload_config_files()
    except SystemExit as exc:
        LOG.warning('Caught SystemExit while reloading configure files with exit code: %d', exc.code)
        return False
    except Error as err:
        LOG.warning('Caught Error while reloading configure files: %s', err)
        return False
    else:
        self._namespace = namespace
        return True
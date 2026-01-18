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
def list_all_sections(self):
    """List all sections from the configuration.

        Returns a sorted list of all section names found in the
        configuration files, whether declared beforehand or not.
        """
    s = set([])
    if self._mutable_ns:
        s |= set(self._mutable_ns._sections())
    if self._namespace:
        s |= set(self._namespace._sections())
    return sorted(s)
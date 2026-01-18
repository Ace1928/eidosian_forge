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
def add_parser_argument(self, container, *args, **kwargs):
    values = []
    if container in self._args_cache:
        values = self._args_cache[container]
    values.append({'args': args, 'kwargs': kwargs})
    self._args_cache[container] = values
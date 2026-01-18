import _collections_abc
import abc
import collections
import base64
import functools
import io
import itertools
import logging
import math
import multiprocessing
from operator import itemgetter, attrgetter
import pickletools
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import weakref
import os
import enum
import typing
from functools import wraps
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
from srsly.cloudpickle import register_pickle_by_value
from srsly.cloudpickle import unregister_pickle_by_value
from srsly.cloudpickle import list_registry_pickle_by_value
from srsly.cloudpickle.cloudpickle import _should_pickle_by_reference
from srsly.cloudpickle.cloudpickle import _make_empty_cell, cell_set
from srsly.cloudpickle.cloudpickle import _extract_class_dict, _whichmodule
from srsly.cloudpickle.cloudpickle import _lookup_module_and_qualname
from .testutils import subprocess_pickle_echo
from .testutils import subprocess_pickle_string
from .testutils import assert_run_python_script
from .testutils import subprocess_worker
def test_builtin_slotmethod(self):
    obj = 1.5
    bound_slotmethod = obj.__repr__
    unbound_slotmethod = type(obj).__repr__
    clsdict_slotmethod = type(obj).__dict__['__repr__']
    depickled_bound_meth = pickle_depickle(bound_slotmethod, protocol=self.protocol)
    depickled_unbound_meth = pickle_depickle(unbound_slotmethod, protocol=self.protocol)
    depickled_clsdict_meth = pickle_depickle(clsdict_slotmethod, protocol=self.protocol)
    assert depickled_bound_meth() == bound_slotmethod()
    assert depickled_unbound_meth is unbound_slotmethod
    assert depickled_clsdict_meth is clsdict_slotmethod
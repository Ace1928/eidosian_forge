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
def test_interactive_remote_function_calls(self):
    code = 'if __name__ == "__main__":\n        from srsly.tests.cloudpickle.testutils import subprocess_worker\n\n        def interactive_function(x):\n            return x + 1\n\n        with subprocess_worker(protocol={protocol}) as w:\n\n            assert w.run(interactive_function, 41) == 42\n\n            # Define a new function that will call an updated version of\n            # the previously called function:\n\n            def wrapper_func(x):\n                return interactive_function(x)\n\n            def interactive_function(x):\n                return x - 1\n\n            # The change in the definition of interactive_function in the main\n            # module of the main process should be reflected transparently\n            # in the worker process: the worker process does not recall the\n            # previous definition of `interactive_function`:\n\n            assert w.run(wrapper_func, 41) == 40\n        '.format(protocol=self.protocol)
    assert_run_python_script(code)
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
def test_interactively_defined_enum(self):
    code = 'if __name__ == "__main__":\n        from enum import Enum\n        from srsly.tests.cloudpickle.testutils import subprocess_worker\n\n        with subprocess_worker(protocol={protocol}) as w:\n\n            class Color(Enum):\n                RED = 1\n                GREEN = 2\n\n            def check_positive(x):\n                return Color.GREEN if x >= 0 else Color.RED\n\n            result = w.run(check_positive, 1)\n\n            # Check that the returned enum instance is reconciled with the\n            # locally defined Color enum type definition:\n\n            assert result is Color.GREEN\n\n            # Check that changing the definition of the Enum class is taken\n            # into account on the worker for subsequent calls:\n\n            class Color(Enum):\n                RED = 1\n                BLUE = 2\n\n            def check_positive(x):\n                return Color.BLUE if x >= 0 else Color.RED\n\n            result = w.run(check_positive, 1)\n            assert result is Color.BLUE\n        '.format(protocol=self.protocol)
    assert_run_python_script(code)
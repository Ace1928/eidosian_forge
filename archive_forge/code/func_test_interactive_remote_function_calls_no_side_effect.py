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
def test_interactive_remote_function_calls_no_side_effect(self):
    code = 'if __name__ == "__main__":\n        from srsly.tests.cloudpickle.testutils import subprocess_worker\n        import sys\n\n        with subprocess_worker(protocol={protocol}) as w:\n\n            GLOBAL_VARIABLE = 0\n\n            class CustomClass(object):\n\n                def mutate_globals(self):\n                    global GLOBAL_VARIABLE\n                    GLOBAL_VARIABLE += 1\n                    return GLOBAL_VARIABLE\n\n            custom_object = CustomClass()\n            assert w.run(custom_object.mutate_globals) == 1\n\n            # The caller global variable is unchanged in the main process.\n\n            assert GLOBAL_VARIABLE == 0\n\n            # Calling the same function again starts again from zero. The\n            # worker process is stateless: it has no memory of the past call:\n\n            assert w.run(custom_object.mutate_globals) == 1\n\n            # The symbols defined in the main process __main__ module are\n            # not set in the worker process main module to leave the worker\n            # as stateless as possible:\n\n            def is_in_main(name):\n                return hasattr(sys.modules["__main__"], name)\n\n            assert is_in_main("CustomClass")\n            assert not w.run(is_in_main, "CustomClass")\n\n            assert is_in_main("GLOBAL_VARIABLE")\n            assert not w.run(is_in_main, "GLOBAL_VARIABLE")\n\n        '.format(protocol=self.protocol)
    assert_run_python_script(code)
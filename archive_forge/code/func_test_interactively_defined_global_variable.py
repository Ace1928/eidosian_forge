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
def test_interactively_defined_global_variable(self):
    code_template = '        from srsly.tests.cloudpickle.testutils import subprocess_pickle_echo\n        from srsly.cloudpickle import dumps, loads\n\n        def local_clone(obj, protocol=None):\n            return loads(dumps(obj, protocol=protocol))\n\n        VARIABLE = "default_value"\n\n        def f0():\n            global VARIABLE\n            VARIABLE = "changed_by_f0"\n\n        def f1():\n            return VARIABLE\n\n        assert f0.__globals__ is f1.__globals__\n\n        # pickle f0 and f1 inside the same pickle_string\n        cloned_f0, cloned_f1 = {clone_func}([f0, f1], protocol={protocol})\n\n        # cloned_f0 and cloned_f1 now share a global namespace that is isolated\n        # from any previously existing namespace\n        assert cloned_f0.__globals__ is cloned_f1.__globals__\n        assert cloned_f0.__globals__ is not f0.__globals__\n\n        # pickle f1 another time, but in a new pickle string\n        pickled_f1 = dumps(f1, protocol={protocol})\n\n        # Change the value of the global variable in f0\'s new global namespace\n        cloned_f0()\n\n        # thanks to cloudpickle isolation, depickling and calling f0 and f1\n        # should not affect the globals of already existing modules\n        assert VARIABLE == "default_value", VARIABLE\n\n        # Ensure that cloned_f1 and cloned_f0 share the same globals, as f1 and\n        # f0 shared the same globals at pickling time, and cloned_f1 was\n        # depickled from the same pickle string as cloned_f0\n        shared_global_var = cloned_f1()\n        assert shared_global_var == "changed_by_f0", shared_global_var\n\n        # f1 is unpickled another time, but because it comes from another\n        # pickle string than pickled_f1 and pickled_f0, it will not share the\n        # same globals as the latter two.\n        new_cloned_f1 = loads(pickled_f1)\n        assert new_cloned_f1.__globals__ is not cloned_f1.__globals__\n        assert new_cloned_f1.__globals__ is not f1.__globals__\n\n        # get the value of new_cloned_f1\'s VARIABLE\n        new_global_var = new_cloned_f1()\n        assert new_global_var == "default_value", new_global_var\n        '
    for clone_func in ['local_clone', 'subprocess_pickle_echo']:
        code = code_template.format(protocol=self.protocol, clone_func=clone_func)
        assert_run_python_script(textwrap.dedent(code))
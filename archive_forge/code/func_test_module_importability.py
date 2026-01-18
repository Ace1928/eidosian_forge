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
def test_module_importability(self):
    pytest.importorskip('_cloudpickle_testpkg')
    from srsly.cloudpickle.compat import pickle
    import os.path
    import collections
    import collections.abc
    assert _should_pickle_by_reference(pickle)
    assert _should_pickle_by_reference(os.path)
    assert _should_pickle_by_reference(collections)
    assert _should_pickle_by_reference(collections.abc)
    dynamic_module = types.ModuleType('dynamic_module')
    assert not _should_pickle_by_reference(dynamic_module)
    if platform.python_implementation() == 'PyPy':
        import _codecs
        assert _should_pickle_by_reference(_codecs)
    import _cloudpickle_testpkg.mod.dynamic_submodule as m
    assert _should_pickle_by_reference(m)
    assert pickle_depickle(m, protocol=self.protocol) is m
    from _cloudpickle_testpkg.mod import dynamic_submodule_two as m2
    assert _should_pickle_by_reference(m2)
    assert pickle_depickle(m2, protocol=self.protocol) is m2
    with pytest.raises(ImportError):
        import _cloudpickle_testpkg.mod.submodule_three
    from _cloudpickle_testpkg.mod import submodule_three as m3
    assert not _should_pickle_by_reference(m3)
    assert not hasattr(m3, '__module__')
    depickled_m3 = pickle_depickle(m3, protocol=self.protocol)
    assert depickled_m3 is not m3
    assert m3.f(1) == depickled_m3.f(1)
    import _cloudpickle_testpkg.mod.dynamic_submodule.dynamic_subsubmodule as sm
    assert _should_pickle_by_reference(sm)
    assert pickle_depickle(sm, protocol=self.protocol) is sm
    expected = 'cannot check importability of object instances'
    with pytest.raises(TypeError, match=expected):
        _should_pickle_by_reference(object())
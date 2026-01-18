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
def test_interactive_dynamic_type_and_stored_remote_instances(self):
    """Simulate objects stored on workers to check isinstance semantics

        Such instances stored in the memory of running worker processes are
        similar to dask-distributed futures for instance.
        """
    code = 'if __name__ == "__main__":\n        import srsly.cloudpickle as cloudpickle, uuid\n        from srsly.tests.cloudpickle.testutils import subprocess_worker\n\n        with subprocess_worker(protocol={protocol}) as w:\n\n            class A:\n                \'\'\'Original class definition\'\'\'\n                pass\n\n            def store(x):\n                storage = getattr(cloudpickle, "_test_storage", None)\n                if storage is None:\n                    storage = cloudpickle._test_storage = dict()\n                obj_id = uuid.uuid4().hex\n                storage[obj_id] = x\n                return obj_id\n\n            def lookup(obj_id):\n                return cloudpickle._test_storage[obj_id]\n\n            id1 = w.run(store, A())\n\n            # The stored object on the worker is matched to a singleton class\n            # definition thanks to provenance tracking:\n            assert w.run(lambda obj_id: isinstance(lookup(obj_id), A), id1)\n\n            # Retrieving the object from the worker yields a local copy that\n            # is matched back the local class definition this instance\n            # originally stems from.\n            assert isinstance(w.run(lookup, id1), A)\n\n            # Changing the local class definition should be taken into account\n            # in all subsequent calls. In particular the old instances on the\n            # worker do not map back to the new class definition, neither on\n            # the worker itself, nor locally on the main program when the old\n            # instance is retrieved:\n\n            class A:\n                \'\'\'Updated class definition\'\'\'\n                pass\n\n            assert not w.run(lambda obj_id: isinstance(lookup(obj_id), A), id1)\n            retrieved1 = w.run(lookup, id1)\n            assert not isinstance(retrieved1, A)\n            assert retrieved1.__class__ is not A\n            assert retrieved1.__class__.__doc__ == "Original class definition"\n\n            # New instances on the other hand are proper instances of the new\n            # class definition everywhere:\n\n            a = A()\n            id2 = w.run(store, a)\n            assert w.run(lambda obj_id: isinstance(lookup(obj_id), A), id2)\n            assert isinstance(w.run(lookup, id2), A)\n\n            # Monkeypatch the class defintion in the main process to a new\n            # class method:\n            A.echo = lambda cls, x: x\n\n            # Calling this method on an instance will automatically update\n            # the remote class definition on the worker to propagate the monkey\n            # patch dynamically.\n            assert w.run(a.echo, 42) == 42\n\n            # The stored instance can therefore also access the new class\n            # method:\n            assert w.run(lambda obj_id: lookup(obj_id).echo(43), id2) == 43\n\n        '.format(protocol=self.protocol)
    assert_run_python_script(code)
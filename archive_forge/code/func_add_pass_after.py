import timeit
from abc import abstractmethod, ABCMeta
from collections import namedtuple, OrderedDict
import inspect
from pprint import pformat
from numba.core.compiler_lock import global_compiler_lock
from numba.core import errors, config, transforms, utils
from numba.core.tracing import event
from numba.core.postproc import PostProcessor
from numba.core.ir_utils import enforce_no_dels, legalize_single_scope
import numba.core.event as ev
def add_pass_after(self, pass_cls, location):
    """
        Add a pass `pass_cls` to the PassManager's compilation pipeline after
        the pass `location`.
        """
    assert self.passes
    self._validate_pass(pass_cls)
    self._validate_pass(location)
    for idx, (x, _) in enumerate(self.passes):
        if x == location:
            break
    else:
        raise ValueError('Could not find pass %s' % location)
    self.passes.insert(idx + 1, (pass_cls, str(pass_cls)))
    self._finalized = False
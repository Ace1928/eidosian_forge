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
def find_by_name(self, class_name):
    assert isinstance(class_name, str)
    for k, v in self._registry.items():
        if v.pass_inst.name == class_name:
            return v
    else:
        raise ValueError('No pass with name %s is registered' % class_name)
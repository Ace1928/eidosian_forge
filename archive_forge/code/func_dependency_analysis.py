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
def dependency_analysis(self):
    """
        Computes dependency analysis
        """
    deps = dict()
    for pss, _ in self.passes:
        x = _pass_registry.get(pss).pass_inst
        au = AnalysisUsage()
        x.get_analysis_usage(au)
        deps[type(x)] = au
    requires_map = dict()
    for k, v in deps.items():
        requires_map[k] = v.get_required_set()

    def resolve_requires(key, rmap):

        def walk(lkey, rmap):
            dep_set = rmap[lkey] if lkey in rmap else set()
            if dep_set:
                for x in dep_set:
                    dep_set |= walk(x, rmap)
                return dep_set
            else:
                return set()
        ret = set()
        for k in key:
            ret |= walk(k, rmap)
        return ret
    dep_chain = dict()
    for k, v in requires_map.items():
        dep_chain[k] = set(v) | resolve_requires(v, requires_map)
    return dep_chain
import contextlib
import functools
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, cast
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.resolvelib import BaseReporter, ResolutionImpossible
from pip._vendor.resolvelib import Resolver as RLResolver
from pip._vendor.resolvelib.structs import DirectedGraph
from pip._internal.cache import WheelCache
from pip._internal.index.package_finder import PackageFinder
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import install_req_extend_extras
from pip._internal.req.req_install import InstallRequirement
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver, InstallRequirementProvider
from pip._internal.resolution.resolvelib.provider import PipProvider
from pip._internal.resolution.resolvelib.reporter import (
from pip._internal.utils.packaging import get_requirement
from .base import Candidate, Requirement
from .factory import Factory
def get_installation_order(self, req_set: RequirementSet) -> List[InstallRequirement]:
    """Get order for installation of requirements in RequirementSet.

        The returned list contains a requirement before another that depends on
        it. This helps ensure that the environment is kept consistent as they
        get installed one-by-one.

        The current implementation creates a topological ordering of the
        dependency graph, giving more weight to packages with less
        or no dependencies, while breaking any cycles in the graph at
        arbitrary points. We make no guarantees about where the cycle
        would be broken, other than it *would* be broken.
        """
    assert self._result is not None, 'must call resolve() first'
    if not req_set.requirements:
        return []
    graph = self._result.graph
    weights = get_topological_weights(graph, set(req_set.requirements.keys()))
    sorted_items = sorted(req_set.requirements.items(), key=functools.partial(_req_set_item_sorter, weights=weights), reverse=True)
    return [ireq for _, ireq in sorted_items]
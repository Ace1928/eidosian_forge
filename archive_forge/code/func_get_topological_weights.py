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
def get_topological_weights(graph: 'DirectedGraph[Optional[str]]', requirement_keys: Set[str]) -> Dict[Optional[str], int]:
    """Assign weights to each node based on how "deep" they are.

    This implementation may change at any point in the future without prior
    notice.

    We first simplify the dependency graph by pruning any leaves and giving them
    the highest weight: a package without any dependencies should be installed
    first. This is done again and again in the same way, giving ever less weight
    to the newly found leaves. The loop stops when no leaves are left: all
    remaining packages have at least one dependency left in the graph.

    Then we continue with the remaining graph, by taking the length for the
    longest path to any node from root, ignoring any paths that contain a single
    node twice (i.e. cycles). This is done through a depth-first search through
    the graph, while keeping track of the path to the node.

    Cycles in the graph result would result in node being revisited while also
    being on its own path. In this case, take no action. This helps ensure we
    don't get stuck in a cycle.

    When assigning weight, the longer path (i.e. larger length) is preferred.

    We are only interested in the weights of packages that are in the
    requirement_keys.
    """
    path: Set[Optional[str]] = set()
    weights: Dict[Optional[str], int] = {}

    def visit(node: Optional[str]) -> None:
        if node in path:
            return
        path.add(node)
        for child in graph.iter_children(node):
            visit(child)
        path.remove(node)
        if node not in requirement_keys:
            return
        last_known_parent_count = weights.get(node, 0)
        weights[node] = max(last_known_parent_count, len(path))
    while True:
        leaves = set()
        for key in graph:
            if key is None:
                continue
            for _child in graph.iter_children(key):
                break
            else:
                leaves.add(key)
        if not leaves:
            break
        weight = len(graph) - 1
        for leaf in leaves:
            if leaf not in requirement_keys:
                continue
            weights[leaf] = weight
        for leaf in leaves:
            graph.remove(leaf)
    visit(None)
    difference = set(weights.keys()).difference(requirement_keys)
    assert not difference, difference
    return weights
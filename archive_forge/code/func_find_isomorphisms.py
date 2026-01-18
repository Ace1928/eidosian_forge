import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def find_isomorphisms(self, symmetry=True):
    """Find all subgraph isomorphisms between subgraph and graph

        Finds isomorphisms where :attr:`subgraph` <= :attr:`graph`.

        Parameters
        ----------
        symmetry: bool
            Whether symmetry should be taken into account. If False, found
            isomorphisms may be symmetrically equivalent.

        Yields
        ------
        dict
            The found isomorphism mappings of {graph_node: subgraph_node}.
        """
    if not self.subgraph:
        yield {}
        return
    elif not self.graph:
        return
    elif len(self.graph) < len(self.subgraph):
        return
    if symmetry:
        _, cosets = self.analyze_symmetry(self.subgraph, self._sgn_partitions, self._sge_colors)
        constraints = self._make_constraints(cosets)
    else:
        constraints = []
    candidates = self._find_nodecolor_candidates()
    la_candidates = self._get_lookahead_candidates()
    for sgn in self.subgraph:
        extra_candidates = la_candidates[sgn]
        if extra_candidates:
            candidates[sgn] = candidates[sgn] | {frozenset(extra_candidates)}
    if any(candidates.values()):
        start_sgn = min(candidates, key=lambda n: min(candidates[n], key=len))
        candidates[start_sgn] = (intersect(candidates[start_sgn]),)
        yield from self._map_nodes(start_sgn, candidates, constraints)
    else:
        return
from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from qiskit.exceptions import QiskitError
from .utilities import EPSILON
def manual_get_vertex(polytope, seed=42):
    """
    Returns a single random vertex from `polytope`.
    """
    rng = np.random.default_rng(seed)
    if isinstance(polytope, PolytopeData):
        paragraphs = copy(polytope.convex_subpolytopes)
    elif isinstance(polytope, ConvexPolytopeData):
        paragraphs = [polytope]
    else:
        raise TypeError(f'{type(polytope)} is not polytope-like.')
    rng.shuffle(paragraphs)
    for convex_subpolytope in paragraphs:
        sentences = convex_subpolytope.inequalities + convex_subpolytope.equalities
        if len(sentences) == 0:
            continue
        dimension = len(sentences[0]) - 1
        rng.shuffle(sentences)
        for inequalities in combinations(sentences, dimension):
            matrix = np.array([x[1:] for x in inequalities])
            b = np.array([x[0] for x in inequalities])
            try:
                vertex = np.linalg.inv(-matrix) @ b
                if polytope_has_element(convex_subpolytope, vertex):
                    return vertex
            except np.linalg.LinAlgError:
                pass
    raise QiskitError(f'Polytope has no feasible solutions:\n{polytope}')
from __future__ import annotations
import itertools
import logging
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from monty.json import MSONable, jsanitize
from networkx.algorithms.components import is_connected
from networkx.algorithms.traversal import bfs_tree
from pymatgen.analysis.chemenv.connectivity.environment_nodes import EnvironmentNode
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.graph_utils import get_delta
from pymatgen.analysis.chemenv.utils.math_utils import get_linearly_independent_vectors
def draw_network(env_graph, pos, ax, sg=None, periodicity_vectors=None):
    """Draw network of environments in a matplotlib figure axes.

    Args:
        env_graph: Graph of environments.
        pos: Positions of the nodes of the environments in the 2D figure.
        ax: Axes object in which the network should be drawn.
        sg: Not used currently (drawing of supergraphs).
        periodicity_vectors: List of periodicity vectors that should be drawn.
    """
    for n in env_graph:
        c = Circle(pos[n], radius=0.02, alpha=0.5)
        ax.add_patch(c)
        env_graph.node[n]['patch'] = c
        _x, _y = pos[n]
        ax.annotate(str(n), pos[n], ha='center', va='center', xycoords='data')
    seen = {}
    e = None
    for u, v, d in env_graph.edges(data=True):
        n1 = env_graph.node[u]['patch']
        n2 = env_graph.node[v]['patch']
        rad = 0.1
        if (u, v) in seen:
            rad = seen.get((u, v))
            rad = (rad + np.sign(rad) * 0.1) * -1
        alpha = 0.5
        color = 'k'
        periodic_color = 'r'
        delta = get_delta(u, v, d)
        n1center = np.array(n1.center)
        n2center = np.array(n2.center)
        midpoint = (n1center + n2center) / 2
        dist = np.sqrt(np.power(n2.center[0] - n1.center[0], 2) + np.power(n2.center[1] - n1.center[1], 2))
        n1c_to_n2c = n2center - n1center
        vv = np.cross(np.array([n1c_to_n2c[0], n1c_to_n2c[1], 0], float), np.array([0, 0, 1], float))
        vv /= np.linalg.norm(vv)
        mid_arc = midpoint + rad * dist * np.array([vv[0], vv[1]], float)
        xy_text_offset = 0.1 * dist * np.array([vv[0], vv[1]], float)
        if periodicity_vectors is not None and len(periodicity_vectors) == 1:
            if np.all(np.array(delta) == np.array(periodicity_vectors[0])) or np.all(np.array(delta) == -np.array(periodicity_vectors[0])):
                e = FancyArrowPatch(n1center, n2center, patchA=n1, patchB=n2, arrowstyle='-|>', connectionstyle=f'arc3,rad={rad!r}', mutation_scale=15.0, lw=2, alpha=alpha, color='r', linestyle='dashed')
            else:
                e = FancyArrowPatch(n1center, n2center, patchA=n1, patchB=n2, arrowstyle='-|>', connectionstyle=f'arc3,rad={rad!r}', mutation_scale=10.0, lw=2, alpha=alpha, color=color)
        else:
            ecolor = color if np.allclose(delta, np.zeros(3)) else periodic_color
            e = FancyArrowPatch(n1center, n2center, patchA=n1, patchB=n2, arrowstyle='-|>', connectionstyle=f'arc3,rad={rad!r}', mutation_scale=10.0, lw=2, alpha=alpha, color=ecolor)
        ax.annotate(delta, mid_arc, ha='center', va='center', xycoords='data', xytext=xy_text_offset, textcoords='offset points')
        seen[u, v] = rad
        ax.add_patch(e)
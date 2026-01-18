import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def draw_placements(big_graph: nx.Graph, small_graph: nx.Graph, small_to_big_mappings: Sequence[Dict], max_plots: int=20, axes: Optional[Sequence[plt.Axes]]=None, tilted: bool=True, bad_placement_callback: Optional[Callable[[plt.Axes, int], None]]=None):
    """Draw a visualization of placements from small_graph onto big_graph using Matplotlib.

    The entire `big_graph` will be drawn with default blue colored nodes. `small_graph` nodes
    and edges will be highlighted with a red color.

    Args:
        big_graph: A larger graph to draw with blue colored nodes.
        small_graph: A smaller, sub-graph to highlight with red nodes and edges.
        small_to_big_mappings: A sequence of mappings from `small_graph` nodes to `big_graph`
            nodes.
        max_plots: To prevent an explosion of open Matplotlib figures, we only show the first
            `max_plots` plots.
        axes: Optional list of matplotlib Axes to contain the drawings.
        tilted: Whether to draw gridlike graphs in the ordinary cartesian or tilted plane.
        bad_placement_callback: If provided, we check that the given mappings are valid. If not,
            this callback is called. The callback should accept `ax` and `i` keyword arguments
            for the current axis and mapping index, respectively.
    """
    if len(small_to_big_mappings) > max_plots:
        warnings.warn(f"You've provided a lot of mappings. Only plotting the first {max_plots}")
        small_to_big_mappings = small_to_big_mappings[:max_plots]
    call_show = False
    if axes is None:
        call_show = True
    for i, small_to_big_map in enumerate(small_to_big_mappings):
        if axes is not None:
            ax = axes[i]
        else:
            ax = plt.gca()
        small_mapped = nx.relabel_nodes(small_graph, small_to_big_map)
        if bad_placement_callback is not None:
            if not _is_valid_placement_helper(big_graph=big_graph, small_mapped=small_mapped, small_to_big_mapping=small_to_big_map):
                bad_placement_callback(ax, i)
        draw_gridlike(big_graph, ax=ax, tilted=tilted)
        draw_gridlike(small_mapped, node_color='red', edge_color='red', width=2, with_labels=False, ax=ax, tilted=tilted)
        ax.axis('equal')
        if call_show:
            plt.show()
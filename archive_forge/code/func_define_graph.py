from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
def define_graph(self, graph, force=False):
    """
        Declares a custom graph structure for the Pipeline overriding
        the default linear flow. The graph should be defined as an
        adjacency mapping.

        Arguments
        ---------
        graph: dict
          Dictionary declaring the relationship between different
          pipeline stages. Should map from a single stage name to
          one or more stage names.
        """
    stages = list(self._stages)
    if not stages:
        self._graph = {}
        return
    graph = {k: v if isinstance(v, tuple) else (v,) for k, v in graph.items()}
    not_found = []
    for source, targets in graph.items():
        if source not in stages:
            not_found.append(source)
        not_found += [t for t in targets if t not in stages]
    if not_found:
        raise ValueError('Pipeline stage(s) %s not found, ensure all stages referenced in the graph have been added.' % (not_found[0] if len(not_found) == 1 else not_found))
    if graph:
        if not (self._linear or force):
            raise ValueError('Graph has already been defined, cannot override existing graph.')
        self._linear = False
    else:
        graph = {s: (t,) for s, t in zip(stages[:-1], stages[1:])}
    root = get_root(graph)
    if not is_traversable(root, graph, stages):
        raise ValueError('Graph is not fully traversable from stage: %s.' % root)
    reinit = root is not self._stage
    self._stage = root
    self._graph = graph
    self._route = [root]
    if not self._linear:
        self.buttons[:] = [Column(self.prev_selector, self.prev_button), Column(self.next_selector, self.next_button)]
    if reinit:
        self.stage[:] = [self._init_stage()]
    self._update_progress()
    self._update_button()
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
def is_traversable(root, graph, stages):
    """
    Check if the graph is fully traversable from the root node.
    """
    int_graph = {stages.index(s): tuple((stages.index(t) for t in tgts)) for s, tgts in graph.items()}
    visited = [False] * len(stages)
    traverse(int_graph, stages.index(root), visited)
    return all(visited)
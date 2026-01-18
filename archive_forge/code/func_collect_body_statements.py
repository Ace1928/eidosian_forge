from __future__ import annotations
import ast
import contextlib
import re
import textwrap
import traceback
from typing import Any, Iterable
from streamlit.runtime.metrics_util import gather_metrics
def collect_body_statements(node: ast.AST) -> None:
    if not hasattr(node, 'body'):
        return
    for child in ast.iter_child_nodes(node):
        if hasattr(child, 'lineno'):
            line_to_node_map[child.lineno] = child
            collect_body_statements(child)
import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
def render_node(self, k: str, node: GraphNode):
    if node.kind == 'valuestate':
        self.digraph.node(k, label=node.data['body'], shape='rect')
    elif node.kind == 'op':
        self.digraph.node(k, label=node.data['body'], shape='box', style='rounded')
    elif node.kind == 'effect':
        self.digraph.node(k, label=node.data['body'], shape='circle')
    elif node.kind == 'meta':
        self.digraph.node(k, label=node.data['body'], shape='plain', fontcolor='grey')
    elif node.kind == 'ports':
        ports = [f'<{x}> {x}' for x in node.ports]
        label = f'{node.data['body']} | {'|'.join(ports)}'
        self.digraph.node(k, label=label, shape='record')
    elif node.kind == 'cfg':
        self.digraph.node(k, label=node.data['body'], shape='plain', fontcolor='blue')
    else:
        self.digraph.node(k, label=f'{k}\n{node.kind}\n{node.data.get('body', '')}', shape='rect')
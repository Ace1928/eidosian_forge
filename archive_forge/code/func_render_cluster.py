import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
@contextmanager
def render_cluster(self, name: str):
    with self.digraph.subgraph(name=f'cluster_{name}') as subg:
        attrs = dict(color='black', bgcolor='white')
        if name.startswith('regionouter'):
            attrs['bgcolor'] = 'grey'
        elif name.startswith('loop_'):
            attrs['color'] = 'blue'
        elif name.startswith('switch_'):
            attrs['color'] = 'green'
        subg.attr(**attrs)
        yield type(self)(subg)
import typing as t
from . import nodes
from .visitor import NodeVisitor
def dump_stores(self) -> t.Dict[str, str]:
    rv: t.Dict[str, str] = {}
    node: t.Optional['Symbols'] = self
    while node is not None:
        for name in sorted(node.stores):
            if name not in rv:
                rv[name] = self.find_ref(name)
        node = node.parent
    return rv
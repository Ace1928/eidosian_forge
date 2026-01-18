import typing as t
from . import nodes
from .visitor import NodeVisitor
def find_ref(self, name: str) -> t.Optional[str]:
    if name in self.refs:
        return self.refs[name]
    if self.parent is not None:
        return self.parent.find_ref(name)
    return None
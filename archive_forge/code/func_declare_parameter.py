import typing as t
from . import nodes
from .visitor import NodeVisitor
def declare_parameter(self, name: str) -> str:
    self.stores.add(name)
    return self._define_ref(name, load=(VAR_LOAD_PARAMETER, None))
import typing as t
from . import nodes
from .visitor import NodeVisitor
def symbols_for_node(node: nodes.Node, parent_symbols: t.Optional['Symbols']=None) -> 'Symbols':
    sym = Symbols(parent=parent_symbols)
    sym.analyze_node(node)
    return sym
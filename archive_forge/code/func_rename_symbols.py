import ast
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
def rename_symbols(node, name_map):
    """Renames symbols in an AST. Requires qual_names annotations."""
    renamer = SymbolRenamer(name_map)
    if isinstance(node, list):
        return [renamer.visit(n) for n in node]
    elif isinstance(node, tuple):
        return tuple((renamer.visit(n) for n in node))
    return renamer.visit(node)
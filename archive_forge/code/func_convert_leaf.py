from parso.python import tree
from parso.python.token import PythonTokenTypes
from parso.parser import BaseParser
def convert_leaf(self, type, value, prefix, start_pos):
    if type == NAME:
        if value in self._pgen_grammar.reserved_syntax_strings:
            return tree.Keyword(value, start_pos, prefix)
        else:
            return tree.Name(value, start_pos, prefix)
    return self._leaf_map.get(type, tree.Operator)(value, start_pos, prefix)
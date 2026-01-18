from typing import Iterable, List, Tuple
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.tree.Tree import TerminalNode, Token, Tree
from _qpd_antlr import QPDLexer, QPDParser
@property
def raw_code(self):
    return self._raw_code
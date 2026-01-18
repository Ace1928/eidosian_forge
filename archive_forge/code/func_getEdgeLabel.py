from io import StringIO
from antlr4 import DFA
from antlr4.Utils import str_list
from antlr4.dfa.DFAState import DFAState
def getEdgeLabel(self, i: int):
    return "'" + chr(i) + "'"
from io import StringIO
from antlr4 import Parser, DFA
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.error.ErrorListener import ErrorListener
def reportAmbiguity(self, recognizer: Parser, dfa: DFA, startIndex: int, stopIndex: int, exact: bool, ambigAlts: set, configs: ATNConfigSet):
    if self.exactOnly and (not exact):
        return
    with StringIO() as buf:
        buf.write('reportAmbiguity d=')
        buf.write(self.getDecisionDescription(recognizer, dfa))
        buf.write(': ambigAlts=')
        buf.write(str(self.getConflictingAlts(ambigAlts, configs)))
        buf.write(", input='")
        buf.write(recognizer.getTokenStream().getText(startIndex, stopIndex))
        buf.write("'")
        recognizer.notifyErrorListeners(buf.getvalue())
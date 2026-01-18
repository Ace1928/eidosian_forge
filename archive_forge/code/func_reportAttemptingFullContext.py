from io import StringIO
from antlr4 import Parser, DFA
from antlr4.atn.ATNConfigSet import ATNConfigSet
from antlr4.error.ErrorListener import ErrorListener
def reportAttemptingFullContext(self, recognizer: Parser, dfa: DFA, startIndex: int, stopIndex: int, conflictingAlts: set, configs: ATNConfigSet):
    with StringIO() as buf:
        buf.write('reportAttemptingFullContext d=')
        buf.write(self.getDecisionDescription(recognizer, dfa))
        buf.write(", input='")
        buf.write(recognizer.getTokenStream().getText(startIndex, stopIndex))
        buf.write("'")
        recognizer.notifyErrorListeners(buf.getvalue())
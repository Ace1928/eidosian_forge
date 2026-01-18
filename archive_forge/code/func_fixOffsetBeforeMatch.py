from antlr4.InputStream import InputStream
from antlr4.atn.LexerAction import LexerAction, LexerIndexedCustomAction
def fixOffsetBeforeMatch(self, offset: int):
    updatedLexerActions = None
    for i in range(0, len(self.lexerActions)):
        if self.lexerActions[i].isPositionDependent and (not isinstance(self.lexerActions[i], LexerIndexedCustomAction)):
            if updatedLexerActions is None:
                updatedLexerActions = [la for la in self.lexerActions]
            updatedLexerActions[i] = LexerIndexedCustomAction(offset, self.lexerActions[i])
    if updatedLexerActions is None:
        return self
    else:
        return LexerActionExecutor(updatedLexerActions)
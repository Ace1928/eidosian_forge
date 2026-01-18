from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.error.ErrorListener import ProxyErrorListener, ConsoleErrorListener
def getRuleIndexMap(self):
    ruleNames = self.getRuleNames()
    if ruleNames is None:
        from antlr4.error.Errors import UnsupportedOperationException
        raise UnsupportedOperationException('The current recognizer does not provide a list of rule names.')
    result = self.ruleIndexMapCache.get(ruleNames, None)
    if result is None:
        result = zip(ruleNames, range(0, len(ruleNames)))
        self.ruleIndexMapCache[ruleNames] = result
    return result
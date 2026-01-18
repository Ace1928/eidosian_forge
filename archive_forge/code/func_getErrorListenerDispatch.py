from antlr4.RuleContext import RuleContext
from antlr4.Token import Token
from antlr4.error.ErrorListener import ProxyErrorListener, ConsoleErrorListener
def getErrorListenerDispatch(self):
    return ProxyErrorListener(self._listeners)
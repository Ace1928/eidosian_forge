import re
from IPython.core.hooks import CommandChainDispatcher
def add_re(self, regex, obj, priority=0):
    """ Adds a target regexp for dispatching """
    chain = self.regexs.get(regex, CommandChainDispatcher())
    chain.add(obj, priority)
    self.regexs[regex] = chain
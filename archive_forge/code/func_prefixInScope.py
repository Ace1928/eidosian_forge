from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def prefixInScope(self, prefix):
    stack = self.prefixStack
    for i in range(-1, (len(self.prefixStack) + 1) * -1, -1):
        if prefix in stack[i]:
            return True
    return False
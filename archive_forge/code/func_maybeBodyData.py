from twisted.internet.protocol import Protocol
from twisted.python.reflect import prefixedMethodNames
def maybeBodyData(self):
    if self.endtag:
        return 'bodydata'
    if self.tagName == 'script' and 'src' not in self.tagAttributes:
        self.begin_bodydata(None)
        return 'waitforendscript'
    return 'bodydata'
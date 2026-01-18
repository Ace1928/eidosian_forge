from ._constants import *
def getuntil(self, terminator, name):
    result = ''
    while True:
        c = self.next
        self.__next()
        if c is None:
            if not result:
                raise self.error('missing ' + name)
            raise self.error('missing %s, unterminated name' % terminator, len(result))
        if c == terminator:
            if not result:
                raise self.error('missing ' + name, 1)
            break
        result += c
    return result
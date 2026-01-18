from ._constants import *
def checkgroupname(self, name, offset, nested):
    if not name.isidentifier():
        msg = 'bad character in group name %r' % name
        raise self.error(msg, len(name) + offset)
    if not (self.istext or name.isascii()):
        import warnings
        warnings.warn('bad character in group name %a at position %d' % (name, self.tell() - len(name) - offset), DeprecationWarning, stacklevel=nested + 7)
from fontTools.cffLib import maxStackLimit
@staticmethod
def rlineto(args):
    if not args:
        raise ValueError(args)
    for args in _everyN(args, 2):
        yield ('rlineto', args)
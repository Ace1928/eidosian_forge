from fontTools.cffLib import maxStackLimit
@staticmethod
def hmoveto(args):
    if len(args) != 1:
        raise ValueError(args)
    yield ('rmoveto', [args[0], 0])
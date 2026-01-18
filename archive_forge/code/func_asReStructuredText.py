import zope.interface
def asReStructuredText(I, munge=0):
    """ Output reStructuredText format.  Note, this will whack any existing
    'structured' format of the text."""
    return asStructuredText(I, munge=munge, rst=True)
from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def elementStream():
    """Preferred method to construct an ElementStream

    Uses Expat-based stream if available, and falls back to Sux if necessary.
    """
    try:
        es = ExpatElementStream()
        return es
    except ImportError:
        if SuxElementStream is None:
            raise Exception('No parsers available :(')
        es = SuxElementStream()
        return es
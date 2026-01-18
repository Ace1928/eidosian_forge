from __future__ import unicode_literals
from os import path
from pybtex import Engine
def make_bibliography(*args, **kwargs):
    """A convenience function that calls :py:meth:`.BibTeXEngine.make_bibliography`."""
    return BibTeXEngine().make_bibliography(*args, **kwargs)
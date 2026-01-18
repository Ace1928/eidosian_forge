from __future__ import unicode_literals
from os import path
from pybtex import Engine
def format_from_string(*args, **kwargs):
    """A convenience function that calls :py:meth:`.BibTeXEngine.format_from_string`."""
    return BibTeXEngine().format_from_string(*args, **kwargs)
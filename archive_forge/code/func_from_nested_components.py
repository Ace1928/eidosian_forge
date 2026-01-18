import re
import pydoc
from .external.docscrape import NumpyDocString
@classmethod
def from_nested_components(cls, **kwargs):
    """Add multiple sub-sets of components."""
    return cls(kwargs, strip_whitespace=False)
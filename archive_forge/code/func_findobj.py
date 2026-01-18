from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def findobj(self, match=None, include_self=True):
    """
        Find artist objects.

        Recursively find all `.Artist` instances contained in the artist.

        Parameters
        ----------
        match
            A filter criterion for the matches. This can be

            - *None*: Return all objects contained in artist.
            - A function with signature ``def match(artist: Artist) -> bool``.
              The result will only contain artists for which the function
              returns *True*.
            - A class instance: e.g., `.Line2D`. The result will only contain
              artists of this class or its subclasses (``isinstance`` check).

        include_self : bool
            Include *self* in the list to be checked for a match.

        Returns
        -------
        list of `.Artist`

        """
    if match is None:

        def matchfunc(x):
            return True
    elif isinstance(match, type) and issubclass(match, Artist):

        def matchfunc(x):
            return isinstance(x, match)
    elif callable(match):
        matchfunc = match
    else:
        raise ValueError('match must be None, a matplotlib.artist.Artist subclass, or a callable')
    artists = sum([c.findobj(matchfunc) for c in self.get_children()], [])
    if include_self and matchfunc(self):
        artists.append(self)
    return artists
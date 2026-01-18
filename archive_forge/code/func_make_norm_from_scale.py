import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def make_norm_from_scale(scale_cls, base_norm_cls=None, *, init=None):
    """
    Decorator for building a `.Normalize` subclass from a `~.scale.ScaleBase`
    subclass.

    After ::

        @make_norm_from_scale(scale_cls)
        class norm_cls(Normalize):
            ...

    *norm_cls* is filled with methods so that normalization computations are
    forwarded to *scale_cls* (i.e., *scale_cls* is the scale that would be used
    for the colorbar of a mappable normalized with *norm_cls*).

    If *init* is not passed, then the constructor signature of *norm_cls*
    will be ``norm_cls(vmin=None, vmax=None, clip=False)``; these three
    parameters will be forwarded to the base class (``Normalize.__init__``),
    and a *scale_cls* object will be initialized with no arguments (other than
    a dummy axis).

    If the *scale_cls* constructor takes additional parameters, then *init*
    should be passed to `make_norm_from_scale`.  It is a callable which is
    *only* used for its signature.  First, this signature will become the
    signature of *norm_cls*.  Second, the *norm_cls* constructor will bind the
    parameters passed to it using this signature, extract the bound *vmin*,
    *vmax*, and *clip* values, pass those to ``Normalize.__init__``, and
    forward the remaining bound values (including any defaults defined by the
    signature) to the *scale_cls* constructor.
    """
    if base_norm_cls is None:
        return functools.partial(make_norm_from_scale, scale_cls, init=init)
    if isinstance(scale_cls, functools.partial):
        scale_args = scale_cls.args
        scale_kwargs_items = tuple(scale_cls.keywords.items())
        scale_cls = scale_cls.func
    else:
        scale_args = scale_kwargs_items = ()
    if init is None:

        def init(vmin=None, vmax=None, clip=False):
            pass
    return _make_norm_from_scale(scale_cls, scale_args, scale_kwargs_items, base_norm_cls, inspect.signature(init))
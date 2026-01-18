import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
@classmethod
def add_style_opts(cls, component, new_options, backend=None):
    """
        Given a component such as an Element (e.g. Image, Curve) or a
        container (e.g. Layout) specify new style options to be
        accepted by the corresponding plotting class.

        Note: This is supplied for advanced users who know which
        additional style keywords are appropriate for the
        corresponding plotting class.
        """
    backend = cls.current_backend if backend is None else backend
    if component not in cls.registry[backend]:
        raise ValueError(f'Component {component!r} not registered to a plotting class.')
    if not isinstance(new_options, list) or not all((isinstance(el, str) for el in new_options)):
        raise ValueError('Please supply a list of style option keyword strings')
    with param.logging_level('CRITICAL'):
        for option in new_options:
            if option not in cls.registry[backend][component].style_opts:
                plot_class = cls.registry[backend][component]
                plot_class.style_opts = sorted(plot_class.style_opts + [option])
    cls._options[backend][component.name] = Options('style', merge_keywords=True, allowed_keywords=new_options)
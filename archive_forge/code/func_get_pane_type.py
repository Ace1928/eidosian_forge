from __future__ import annotations
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ImportedStyleSheet
from bokeh.models.layouts import (
from .._param import Margin
from ..io.cache import _generate_hash
from ..io.document import create_doc_if_none_exists, unlocked
from ..io.notebook import push
from ..io.state import state
from ..layout.base import (
from ..links import Link
from ..models import ReactiveHTML as _BkReactiveHTML
from ..reactive import Reactive
from ..util import param_reprs, param_watchers
from ..util.checks import is_dataframe, is_series
from ..util.parameters import get_params_to_inherit
from ..viewable import (
@classmethod
def get_pane_type(cls, obj: Any, **kwargs) -> Type['PaneBase']:
    """
        Returns the applicable Pane type given an object by resolving
        the precedence of all types whose applies method declares that
        the object is supported.

        Arguments
        ---------
        obj (object): The object type to return a Pane type for

        Returns
        -------
        The applicable Pane type with the highest precedence.
        """
    if isinstance(obj, Viewable):
        return type(obj)
    descendents = []
    for p in param.concrete_descendents(PaneBase).values():
        if p.priority is None:
            applies = True
            try:
                priority = p.applies(obj, **kwargs if p._applies_kw else {})
            except Exception:
                priority = False
        else:
            applies = None
            priority = p.priority
        if isinstance(priority, bool) and priority:
            raise ValueError('If a Pane declares no priority the applies method should return a priority value specific to the object type or False, but the %s pane declares no priority.' % p.__name__)
        elif priority is None or priority is False:
            continue
        descendents.append((priority, applies, p))
    pane_types = reversed(sorted(descendents, key=lambda x: x[0]))
    for _, applies, pane_type in pane_types:
        if applies is None:
            try:
                applies = pane_type.applies(obj, **kwargs if pane_type._applies_kw else {})
            except Exception:
                applies = False
        if not applies:
            continue
        return pane_type
    raise TypeError('%s type could not be rendered.' % type(obj).__name__)
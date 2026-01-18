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
def collapse_element(cls, overlay, ranges=None, mode='data', backend=None):
    """
        Finds any applicable compositor and applies it.
        """
    from .element import Element
    from .overlay import CompositeOverlay, Overlay
    unpack = False
    if not isinstance(overlay, CompositeOverlay):
        overlay = Overlay([overlay])
        unpack = True
    prev_ids = ()
    processed = defaultdict(list)
    while True:
        match = cls.strongest_match(overlay, mode, backend)
        if match is None:
            if unpack and len(overlay) == 1:
                return overlay.values()[0]
            return overlay
        _, applicable_op, (start, stop) = match
        if isinstance(overlay, Overlay):
            values = overlay.values()
            sliced = Overlay(values[start:stop])
        else:
            values = overlay.items()
            sliced = overlay.clone(values[start:stop])
        items = sliced.traverse(lambda x: x, [Element])
        if applicable_op and all((el in processed[applicable_op] for el in items)):
            if unpack and len(overlay) == 1:
                return overlay.values()[0]
            return overlay
        result = applicable_op.apply(sliced, ranges, backend)
        if applicable_op.group:
            result = result.relabel(group=applicable_op.group)
        if isinstance(overlay, Overlay):
            result = [result]
        else:
            result = list(zip(sliced.keys(), [result]))
        processed[applicable_op] += [el for r in result for el in r.traverse(lambda x: x, [Element])]
        overlay = overlay.clone(values[:start] + result + values[stop:])
        spec_fn = lambda x: not isinstance(x, CompositeOverlay)
        new_ids = tuple(overlay.traverse(lambda x: id(x), [spec_fn]))
        if new_ids == prev_ids:
            return overlay
        prev_ids = new_ids
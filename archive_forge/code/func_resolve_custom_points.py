import re
import warnings
from typing import Dict
import numpy as np
import ase  # Annotations
from ase.utils import jsonable
from ase.cell import Cell
def resolve_custom_points(pathspec, special_points, eps):
    """Resolve a path specification into a string.

    The path specification is a list path segments, each segment being a kpoint
    label or kpoint coordinate, or a single such segment.

    Return a string representing the same path.  Generic kpoint labels
    are generated dynamically as necessary, updating the special_point
    dictionary if necessary.  The tolerance eps is used to see whether
    coordinates are close enough to a special point to deserve being
    labelled as such."""
    special_points = dict(special_points)
    if len(pathspec) == 0:
        return ('', special_points)
    if isinstance(pathspec, str):
        pathspec = parse_path_string(pathspec)

    def looks_like_single_kpoint(obj):
        if isinstance(obj, str):
            return True
        try:
            arr = np.asarray(obj, float)
        except ValueError:
            return False
        else:
            return arr.shape == (3,)
    for thing in pathspec:
        if looks_like_single_kpoint(thing):
            pathspec = [pathspec]
            break

    def name_generator():
        counter = 0
        while True:
            name = 'Kpt{}'.format(counter)
            yield name
            counter += 1
    custom_names = name_generator()
    labelseq = []
    for subpath in pathspec:
        for kpt in subpath:
            if isinstance(kpt, str):
                if kpt not in special_points:
                    raise KeyError('No kpoint "{}" among "{}"'.format(kpt, ''.join(special_points)))
                labelseq.append(kpt)
                continue
            kpt = np.asarray(kpt, float)
            if not kpt.shape == (3,):
                raise ValueError(f'Not a valid kpoint: {kpt}')
            for key, val in special_points.items():
                if np.abs(kpt - val).max() < eps:
                    labelseq.append(key)
                    break
            else:
                name = next(custom_names)
                while name in special_points:
                    name = next(custom_names)
                special_points[name] = kpt
                labelseq.append(name)
        labelseq.append(',')
    last = labelseq.pop()
    assert last == ','
    return (''.join(labelseq), special_points)
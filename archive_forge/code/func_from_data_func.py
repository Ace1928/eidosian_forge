import copy
import numbers
from collections.abc import MutableMapping
from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from .array_sequence import ArraySequence
@classmethod
def from_data_func(cls, data_func):
    """Creates an instance from a generator function.

        The generator function must yield :class:`TractogramItem` objects.

        Parameters
        ----------
        data_func : generator function yielding :class:`TractogramItem` objects
            Generator function that whenever is called starts yielding
            :class:`TractogramItem` objects that will be used to instantiate a
            :class:`LazyTractogram`.

        Returns
        -------
        lazy_tractogram : :class:`LazyTractogram` object
            New lazy tractogram.
        """
    if not callable(data_func):
        raise TypeError('`data_func` must be a generator function.')
    lazy_tractogram = cls()
    lazy_tractogram._data = data_func
    try:
        first_item = next(data_func())

        def _gen(key):
            return lambda: (t.data_for_streamline[key] for t in data_func())
        data_per_streamline_keys = first_item.data_for_streamline.keys()
        for k in data_per_streamline_keys:
            lazy_tractogram._data_per_streamline[k] = _gen(k)

        def _gen(key):
            return lambda: (t.data_for_points[key] for t in data_func())
        data_per_point_keys = first_item.data_for_points.keys()
        for k in data_per_point_keys:
            lazy_tractogram._data_per_point[k] = _gen(k)
    except StopIteration:
        pass
    return lazy_tractogram
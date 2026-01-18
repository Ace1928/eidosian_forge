from __future__ import annotations
import typing
from abc import ABC
from copy import copy
from warnings import warn
import numpy as np
from .._utils import check_required_aesthetics, groupby_apply
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import X_AESTHETICS, Y_AESTHETICS
@classmethod
def collide(cls, data, params):
    """
        Calculate boundaries of geometry object

        Uses Strategy
        """
    xminmax = ['xmin', 'xmax']
    data, width = cls._collide_setup(data, params)
    if params.get('width', None) is None:
        params['width'] = width
    if params and 'reverse' in params and params['reverse']:
        idx = data.sort_values(['xmin', 'group'], kind='mergesort').index
    else:
        data['-group'] = -data['group']
        idx = data.sort_values(['xmin', '-group'], kind='mergesort').index
        del data['-group']
    data = data.loc[idx, :]
    intervals = data[xminmax].drop_duplicates().to_numpy().flatten()
    intervals = intervals[~np.isnan(intervals)]
    if len(np.unique(intervals)) > 1 and any(np.diff(intervals - intervals.mean()) < -1e-06):
        msg = '{} requires non-overlapping x intervals'
        warn(msg.format(cls.__name__), PlotnineWarning)
    if 'ymax' in data:
        data = groupby_apply(data, 'xmin', cls.strategy, params)
    elif 'y' in data:
        data['ymax'] = data['y']
        data = groupby_apply(data, 'xmin', cls.strategy, params)
        data['y'] = data['ymax']
    else:
        raise PlotnineError('Neither y nor ymax defined')
    return data
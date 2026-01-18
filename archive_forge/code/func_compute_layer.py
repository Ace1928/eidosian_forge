from __future__ import annotations
import typing
from copy import deepcopy
import pandas as pd
from .._utils import (
from .._utils.registry import Register, Registry
from ..exceptions import PlotnineError
from ..layer import layer
from ..mapping import aes
from abc import ABC
@classmethod
def compute_layer(cls, data: pd.DataFrame, params: dict[str, Any], layout: Layout) -> pd.DataFrame:
    """
        Calculate statistics for this layers

        This is the top-most computation method for the
        stat. It does not do any computations, but it
        knows how to verify the data, partition it call the
        next computation method and merge results.

        stats should not override this method.

        Parameters
        ----------
        data :
            Data points for all objects in a layer.
        params :
            Stat parameters
        layout :
            Panel layout information
        """
    check_required_aesthetics(cls.REQUIRED_AES, list(data.columns) + list(params.keys()), cls.__name__)
    data = remove_missing(data, na_rm=params.get('na_rm', False), vars=list(cls.REQUIRED_AES | cls.NON_MISSING_AES), name=cls.__name__, finite=True)

    def fn(pdata):
        """
            Compute function helper
            """
        if len(pdata) == 0:
            return pdata
        pscales = layout.get_scales(pdata['PANEL'].iloc[0])
        return cls.compute_panel(pdata, pscales, **params)
    return groupby_apply(data, 'PANEL', fn)
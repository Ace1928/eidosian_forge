from __future__ import annotations
import typing
from copy import copy
import numpy as np
from ..iapi import panel_ranges
def setup_data(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
        Allow the coordinate system to manipulate the layer data

        Parameters
        ----------
        data :
            Data for alls Layer

        Returns
        -------
        :
            Modified layer data
        """
    return data
from __future__ import annotations
import typing
from contextlib import suppress
import numpy as np
from .._utils import match
from ..exceptions import PlotnineError
from ..iapi import labels_view, layout_details, pos_scales
def set_xy_labels(self, labels: labels_view) -> labels_view:
    """
        Determine x & y axis labels

        Parameters
        ----------
        labels : labels_view
            Labels as specified by the user through the `labs` or
            `ylab` calls.

        Returns
        -------
        out : labels_view
            Modified labels
        """
    labels.x = self.xlabel(labels)
    labels.y = self.ylabel(labels)
    return labels
import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
def next_id(self):
    """Returns the `id` for the new node to be inserted.

        The current implementation returns one more than the maximum `id`.

        Returns
        -------
        id : int
            The `id` of the new node to be inserted.
        """
    return self.max_id + 1
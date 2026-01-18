from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
@classmethod
def from_scatter(cls, element, **kwargs):
    """Returns a Slope element given an element of x/y-coordinates

        Computes the slope and y-intercept from an element containing
        x- and y-coordinates.

        Args:
            element: Element to compute slope from
            kwargs: Keyword arguments to pass to the Slope element

        Returns:
            Slope element
        """
    x, y = (element.dimension_values(i) for i in range(2))
    par = np.polyfit(x, y, 1, full=True)
    gradient = par[0][0]
    y_intercept = par[0][1]
    return cls(gradient, y_intercept, **kwargs)
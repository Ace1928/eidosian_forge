import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def sheet2matrix(self, x, y):
    """
        Convert a point (x,y) in Sheet coordinates to continuous
        matrix coordinates.

        Returns (float_row,float_col), where float_row corresponds to
        y, and float_col to x.

        Valid for scalar or array x and y.

        Note about Bounds For a Sheet with
        BoundingBox(points=((-0.5,-0.5),(0.5,0.5))) and density=3,
        x=-0.5 corresponds to float_col=0.0 and x=0.5 corresponds to
        float_col=3.0.  float_col=3.0 is not inside the matrix
        representing this Sheet, which has the three columns
        (0,1,2). That is, x=-0.5 is inside the BoundingBox but x=0.5
        is outside. Similarly, y=0.5 is inside (at row 0) but y=-0.5
        is outside (at row 3) (it's the other way round for y because
        the matrix row index increases as y decreases).
        """
    xdensity = self.__xdensity
    if isinstance(x, np.ndarray) and x.dtype.kind == 'M' or isinstance(x, datetime_types):
        xdensity = np.timedelta64(int(round(1.0 / xdensity)), self._time_unit)
        float_col = (x - self.lbrt[0]) / xdensity
    else:
        float_col = (x - self.lbrt[0]) * xdensity
    ydensity = self.__ydensity
    if isinstance(y, np.ndarray) and y.dtype.kind == 'M' or isinstance(y, datetime_types):
        ydensity = np.timedelta64(int(round(1.0 / ydensity)), self._time_unit)
        float_row = (self.lbrt[3] - y) / ydensity
    else:
        float_row = (self.lbrt[3] - y) * ydensity
    return (float_row, float_col)
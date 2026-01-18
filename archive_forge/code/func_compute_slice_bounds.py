import itertools
import numpy as np
import pandas as pd
import param
from ..core import Dataset
from ..core.boundingregion import BoundingBox
from ..core.data import PandasInterface, default_datatype
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (
def compute_slice_bounds(slices, scs, shape):
    """
    Given a 2D selection consisting of slices/coordinates, a
    SheetCoordinateSystem and the shape of the array returns a new
    BoundingBox representing the sliced region.
    """
    xidx, yidx = slices
    ys, xs = shape
    l, b, r, t = scs.bounds.lbrt()
    xdensity, ydensity = (scs.xdensity, scs.ydensity)
    xunit = 1.0 / xdensity
    yunit = 1.0 / ydensity
    if isinstance(l, datetime_types):
        xunit = np.timedelta64(int(round(xunit)), scs._time_unit)
    if isinstance(b, datetime_types):
        yunit = np.timedelta64(int(round(yunit)), scs._time_unit)
    if isinstance(xidx, slice):
        l = l if xidx.start is None else max(l, xidx.start)
        r = r if xidx.stop is None else min(r, xidx.stop)
    if isinstance(yidx, slice):
        b = b if yidx.start is None else max(b, yidx.start)
        t = t if yidx.stop is None else min(t, yidx.stop)
    bounds = BoundingBox(points=((l, b), (r, t)))
    slc = Slice(bounds, scs)
    l, b, r, t = slc.compute_bounds(scs).lbrt()
    if not isinstance(xidx, slice):
        if not isinstance(xidx, (list, set)):
            xidx = [xidx]
        if len(xidx) > 1:
            xdensity = xdensity * (float(len(xidx)) / xs)
        ls, rs = ([], [])
        for idx in xidx:
            xc, _ = scs.closest_cell_center(idx, b)
            ls.append(xc - xunit / 2)
            rs.append(xc + xunit / 2)
        l, r = (np.min(ls), np.max(rs))
    elif not isinstance(yidx, slice):
        if not isinstance(yidx, (set, list)):
            yidx = [yidx]
        if len(yidx) > 1:
            ydensity = ydensity * (float(len(yidx)) / ys)
        bs, ts = ([], [])
        for idx in yidx:
            _, yc = scs.closest_cell_center(l, idx)
            bs.append(yc - yunit / 2)
            ts.append(yc + yunit / 2)
        b, t = (np.min(bs), np.max(ts))
    return BoundingBox(points=((l, b), (r, t)))
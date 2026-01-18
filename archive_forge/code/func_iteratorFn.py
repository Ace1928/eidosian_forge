import numpy as np
from ..Qt import QtCore, QtGui, QtWidgets
def iteratorFn(self, data):
    if isinstance(data, list) or isinstance(data, tuple):
        return (lambda d: d.__iter__(), None)
    elif isinstance(data, dict):
        return (lambda d: iter(d.values()), list(map(str, data.keys())))
    elif hasattr(data, 'implements') and data.implements('MetaArray'):
        if data.axisHasColumns(0):
            header = [str(data.columnName(0, i)) for i in range(data.shape[0])]
        elif data.axisHasValues(0):
            header = list(map(str, data.xvals(0)))
        else:
            header = None
        return (self.iterFirstAxis, header)
    elif isinstance(data, np.ndarray):
        return (self.iterFirstAxis, None)
    elif isinstance(data, np.void):
        return (self.iterate, list(map(str, data.dtype.names)))
    elif data is None:
        return (None, None)
    elif np.isscalar(data):
        return (self.iterateScalar, None)
    else:
        msg = "Don't know how to iterate over data type: {!s}".format(type(data))
        raise TypeError(msg)
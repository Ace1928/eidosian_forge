import rpy2.robjects as ro
import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
from rpy2.rinterface import (Sexp,
import numpy  # type: ignore
import warnings
def rpy2py_data_frame(obj):
    o2 = list()
    with conversion.get_conversion().context() as cv:
        for column in rinterface.ListSexpVector(obj):
            if 'factor' in column.rclass:
                levels = column.do_slot('levels')
                column = tuple((None if x is rinterface.NA_Integer else levels[x - 1] for x in column))
            o2.append(cv.rpy2py(column))
    names = obj.do_slot('names')
    if names == rinterface.NULL:
        res = numpy.rec.fromarrays(o2)
    else:
        res = numpy.rec.fromarrays(o2, names=tuple(names))
    return res
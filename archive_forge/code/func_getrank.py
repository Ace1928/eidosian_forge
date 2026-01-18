import typing
from rpy2.rinterface_lib import openrlib
def getrank(cdata) -> int:
    """Get the rank (number of dimensions) of an R array.

    The R NULL will return a rank of 1.
    :param cdata: C data from cffi
    :return: The rank"""
    dim_cdata = openrlib.rlib.Rf_getAttrib(cdata, openrlib.rlib.R_DimSymbol)
    if dim_cdata == openrlib.rlib.R_NilValue:
        return 1
    else:
        return openrlib.rlib.Rf_length(dim_cdata)
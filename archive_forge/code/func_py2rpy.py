from .. import utils
from .._lazyload import anndata2ri
from .._lazyload import rpy2
import numpy as np
import warnings
@utils._with_pkg(pkg='rpy2', min_version='3.0')
def py2rpy(pyobject):
    """Convert an Python object to rpy2.

    Attempts the following, in order: data.frame -> pd.DataFrame, named list -> dict,
    unnamed list -> list, SingleCellExperiment -> anndata.AnnData, vector -> np.ndarray,
    rpy2 generic converter, NULL -> None.

    Parameters
    ----------
    pyobject : python object
        Converted object

    Returns
    -------
    robject : rpy2 object
        Object to be converted
    """
    for converter in [_pynull2rpy, _pysce2rpy, rpy2.robjects.pandas2ri.converter.py2rpy, rpy2.robjects.numpy2ri.converter.py2rpy, rpy2.robjects.default_converter.py2rpy]:
        if not _is_r_object(pyobject):
            try:
                pyobject = converter(pyobject)
            except NotImplementedError:
                pass
        else:
            break
    if not _is_r_object(pyobject) and (not _is_builtin(pyobject)):
        warnings.warn('Object not converted: {} (type {})'.format(pyobject, type(pyobject).__name__), RuntimeWarning)
    return pyobject
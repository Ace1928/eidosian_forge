from . import matrix
from deprecated import deprecated
import numbers
import warnings
def is_Anndata(X):
    try:
        return isinstance(X, anndata.AnnData)
    except NameError:
        return False
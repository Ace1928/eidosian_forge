from . import matrix
from deprecated import deprecated
import numbers
import warnings
def is_DataFrame(X):
    try:
        return isinstance(X, pd.DataFrame)
    except NameError:
        return False
from typing import Iterator, Optional, Tuple
import numpy as np
import pandas
from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, IndexLabel
from pandas.util._decorators import doc
from modin.utils import func_from_deprecated_location, hashable
from_pandas = func_from_deprecated_location(
from_arrow = func_from_deprecated_location(
from_dataframe = func_from_deprecated_location(
from_non_pandas = func_from_deprecated_location(
def check_both_not_none(option1, option2):
    """
    Check that both `option1` and `option2` are not None.

    Parameters
    ----------
    option1 : Any
        First object to check if not None.
    option2 : Any
        Second object to check if not None.

    Returns
    -------
    bool
        True if both option1 and option2 are not None, False otherwise.
    """
    return not (option1 is None or option2 is None)
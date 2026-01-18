import enum
import re
from typing import Optional, Union
import numpy as np
import pandas
from pandas.api.types import is_datetime64_dtype
def raise_copy_alert(copy_reason: Optional[str]=None) -> None:
    """
    Raise a ``RuntimeError`` mentioning that there's a copy required.

    Parameters
    ----------
    copy_reason : str, optional
        The reason of making a copy. Should fit to the following format:
        'The copy occured due to {copy_reason}.'.
    """
    msg = "Copy required but 'allow_copy=False' is set."
    if copy_reason:
        msg += f' The copy occured due to {copy_reason}.'
    raise RuntimeError(msg)
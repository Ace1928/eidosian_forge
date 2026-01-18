import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def execute_callable(fn, inplace=False, md_kwargs={}, pd_kwargs={}):
    try:
        pd_result = fn(pandas_df, **pd_kwargs)
    except Exception as pd_e:
        try:
            if inplace:
                _ = fn(modin_df, **md_kwargs)
                try_cast_to_pandas(modin_df)
            else:
                try_cast_to_pandas(fn(modin_df, **md_kwargs))
        except Exception as md_e:
            assert isinstance(md_e, type(pd_e)), 'Got Modin Exception type {}, but pandas Exception type {} was expected'.format(type(md_e), type(pd_e))
            if expected_exception:
                if Engine.get() == 'Ray':
                    from ray.exceptions import RayTaskError
                    if isinstance(md_e, RayTaskError):
                        md_e = md_e.args[0]
                assert type(md_e) is type(expected_exception) and md_e.args == expected_exception.args, f"not acceptable Modin's exception: [{repr(md_e)}]"
                assert pd_e.args == expected_exception.args, f"not acceptable Pandas' exception: [{repr(pd_e)}]"
            elif expected_exception is False:
                pass
            else:
                raise pd_e
        else:
            raise NoModinException(f"Modin doesn't throw an exception, while pandas does: [{repr(pd_e)}]")
    else:
        md_result = fn(modin_df, **md_kwargs)
        return (md_result, pd_result) if not inplace else (modin_df, pandas_df)
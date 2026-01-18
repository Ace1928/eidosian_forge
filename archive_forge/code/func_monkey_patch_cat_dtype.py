from statsmodels.compat.pandas import PD_LT_2
import pandas as pd
import patsy.util
def monkey_patch_cat_dtype():
    patsy.util.safe_is_pandas_categorical_dtype = _safe_is_pandas_categorical_dtype
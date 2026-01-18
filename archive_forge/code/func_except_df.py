from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def except_df(self, df1: T, df2: T, unique: bool, anti_indicator_col: str=_ANTI_INDICATOR) -> T:
    """Remove df2 from df1

        :param df1: dataframe 1
        :param df2: dataframe 2
        :param unique: whether to remove duplicated rows in the result
        :return: the dataframe with df2 removed
        """
    ndf1, ndf2 = self._preprocess_set_op(df1, df2)
    ndf2 = self._with_indicator(ndf2, anti_indicator_col)
    ndf = ndf1.merge(ndf2, how='left', on=list(ndf1.columns))
    ndf = ndf[ndf[anti_indicator_col].isnull()].drop([anti_indicator_col], axis=1)
    if unique:
        ndf = self.drop_duplicates(ndf)
    return ndf
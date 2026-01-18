from distributed.client import default_client
from modin.core.execution.dask.common import DaskWrapper
from modin.core.execution.dask.implementations.pandas_on_dask.dataframe import (
from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
from modin.core.io import (
from modin.core.storage_formats.pandas.parsers import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.distributed.dataframe.pandas.partitions import (
from modin.experimental.core.io import (
from modin.experimental.core.storage_formats.pandas.parsers import (
from modin.pandas.series import Series
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
def df_to_series(df):
    series = df[df.columns[0]]
    if df.columns[0] == MODIN_UNNAMED_SERIES_LABEL:
        series.name = None
    return series
from functools import wraps
import numpy as np
import pandas
from pandas._libs.lib import no_default
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import is_bool_dtype, is_integer_dtype
from modin.core.storage_formats import BaseQueryCompiler
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.base.query_compiler import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
@wraps(method)
def method_wrapper(self, *args, **kwargs):
    default_method = getattr(super(type(self), self), name, None)
    if is_inoperable([self, args, kwargs]):
        if default_method is None:
            raise NotImplementedError('Frame contains data of unsupported types.')
        return default_method(*args, **kwargs)
    try:
        return method(self, *args, **kwargs)
    except NotImplementedError as err:
        if default_method is None:
            raise err
        ErrorMessage.default_to_pandas(message=str(err))
        return default_method(*args, **kwargs)
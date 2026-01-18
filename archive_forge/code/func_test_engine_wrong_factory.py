from contextlib import contextmanager
import pytest
import modin.pandas as pd
from modin import set_execution
from modin.config import Engine, Parameter, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.dispatcher import (
from modin.core.execution.python.implementations.pandas_on_python.io import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
def test_engine_wrong_factory():
    with pytest.raises(FactoryNotFoundError):
        with _switch_value(Engine, 'Dask'):
            with _switch_value(StorageFormat, 'Pyarrow'):
                pass
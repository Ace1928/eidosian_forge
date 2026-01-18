from contextlib import contextmanager
import pytest
import modin.pandas as pd
from modin import set_execution
from modin.config import Engine, Parameter, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.dispatcher import (
from modin.core.execution.python.implementations.pandas_on_python.io import (
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
def test_add_option():

    class DifferentlyNamedFactory(factories.BaseFactory):

        @classmethod
        def prepare(cls):
            cls.io_cls = PandasOnPythonIO
    factories.StorageOnExecFactory = DifferentlyNamedFactory
    StorageFormat.add_option('sToragE')
    Engine.add_option('Exec')
    with _switch_execution('Exec', 'Storage'):
        df = pd.DataFrame([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        assert isinstance(df._query_compiler, PandasQueryCompiler)
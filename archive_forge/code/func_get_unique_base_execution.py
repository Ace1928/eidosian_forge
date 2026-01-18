import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional
import boto3
import numpy as np
import pandas
import pytest
import requests
import s3fs
from pandas.util._decorators import doc
import modin.utils  # noqa: E402
import uuid  # noqa: E402
import modin  # noqa: E402
import modin.config  # noqa: E402
import modin.tests.config  # noqa: E402
from modin.config import (  # noqa: E402
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
from modin.core.storage_formats import (  # noqa: E402
from modin.tests.pandas.utils import (  # noqa: E402
@pytest.fixture(scope='function')
def get_unique_base_execution():
    """Setup unique execution for a single function and yield its QueryCompiler that's suitable for inplace modifications."""
    execution_id = int(uuid.uuid4().hex, 16)
    format_name = f'Base{execution_id}'
    engine_name = 'Python'
    execution_name = f'{format_name}On{engine_name}'
    base_qc = type(format_name, (TestQC,), {})
    base_io = type(f'{execution_name}IO', (BaseOnPythonIO,), {'query_compiler_cls': base_qc})
    base_factory = type(f'{execution_name}Factory', (BaseOnPythonFactory,), {'prepare': classmethod(lambda cls: setattr(cls, 'io_cls', base_io))})
    setattr(factories, f'{execution_name}Factory', base_factory)
    old_engine, old_format = modin.set_execution(engine=engine_name, storage_format=format_name)
    yield base_qc
    modin.set_execution(engine=old_engine, storage_format=old_format)
    try:
        delattr(factories, f'{execution_name}Factory')
    except AttributeError:
        pass
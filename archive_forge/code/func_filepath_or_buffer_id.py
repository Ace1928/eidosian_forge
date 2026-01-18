from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
@pytest.fixture(params=['string', 'pathlike', 'buffer'])
def filepath_or_buffer_id(request):
    """
    A fixture yielding test ids for filepath_or_buffer testing.
    """
    return request.param
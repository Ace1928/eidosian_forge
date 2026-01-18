from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
@lru_cache()
def python_executable():
    path = shutil.which('python3')
    assert path is not None, "Couldn't find python3 executable"
    return path
import contextlib
import os
import platform
import re
import shutil
import tempfile
from typing import Any, Iterator, List, Mapping, Optional, Tuple, Union
from cmdstanpy import _TMPDIR
from .json import write_stan_json
from .logging import get_logger
def create_named_text_file(dir: str, prefix: str, suffix: str, name_only: bool=False) -> str:
    """
    Create a named unique file, return filename.
    Flag 'name_only' will create then delete the tmp file;
    this lets us create filename args for commands which
    disallow overwriting existing files (e.g., 'stansummary').
    """
    fd = tempfile.NamedTemporaryFile(mode='w+', prefix=prefix, suffix=suffix, dir=dir, delete=name_only)
    path = fd.name
    fd.close()
    return path
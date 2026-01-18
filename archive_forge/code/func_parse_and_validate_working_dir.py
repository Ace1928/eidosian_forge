import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def parse_and_validate_working_dir(working_dir: str) -> str:
    """Parses and validates a 'working_dir' option.

    This should be a URI.
    """
    assert working_dir is not None
    if not isinstance(working_dir, str):
        raise TypeError(f'`working_dir` must be a string, got {type(working_dir)}.')
    validate_uri(working_dir)
    return working_dir
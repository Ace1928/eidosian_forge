import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
def equals(value, default_value):
    if value is None and default_value is None:
        return True
    if value is None or default_value is None:
        return False
    return value == default_value
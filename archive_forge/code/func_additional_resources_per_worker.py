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
@property
def additional_resources_per_worker(self):
    """Resources per worker, not including CPU or GPU resources."""
    return {k: v for k, v in self._resources_per_worker_not_none.items() if k not in ['CPU', 'GPU']}
from __future__ import annotations
import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Set, Tuple, TYPE_CHECKING, Union
from torch._inductor import config
from torch._inductor.utils import get_benchmark_name
@lru_cache
def enabled_metric_tables() -> Set[str]:
    config_str = config.enabled_metric_tables
    enabled = set()
    for name in config_str.split(','):
        name = name.strip()
        if not name:
            continue
        assert name in REGISTERED_METRIC_TABLES, f'Metric table name {name} is not registered'
        enabled.add(name)
    return enabled
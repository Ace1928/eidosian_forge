import dataclasses
import inspect
import io
import pathlib
from dataclasses import dataclass
from typing import List, Type, Dict, Iterator, Tuple, Set
import numpy as np
import pandas as pd
import cirq
from cirq._import import ModuleType
from cirq.protocols.json_serialization import ObjectFactory
def get_resolver_cache_types(self) -> Set[Tuple[str, Type]]:
    result: Set[Tuple[str, Type]] = set()
    for k, v in self.resolver_cache.items():
        if isinstance(v, type):
            result.add((k, v))
    return result
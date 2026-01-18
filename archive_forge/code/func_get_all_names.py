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
def get_all_names(self) -> Iterator[str]:

    def not_module_or_function(x):
        return not (inspect.ismodule(x) or inspect.isfunction(x))
    for m in self.packages:
        for name, _ in inspect.getmembers(m, not_module_or_function):
            yield name
    for name, _ in self.get_resolver_cache_types():
        yield name
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
def spec_for(module_name: str) -> ModuleJsonTestSpec:
    import importlib.util
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f'{module_name} not found')
    test_module_name = f'{module_name}.json_test_data'
    if importlib.util.find_spec(test_module_name) is None:
        raise ValueError(f'{module_name} module is missing json_test_data package, please set it up.')
    test_module = importlib.import_module(test_module_name)
    if not hasattr(test_module, 'TestSpec'):
        raise ValueError(f'{test_module_name} module is missing TestSpec, please set it up.')
    return getattr(test_module, 'TestSpec')
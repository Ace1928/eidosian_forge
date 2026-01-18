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
def all_test_data_keys(self) -> List[str]:
    seen = set()
    for file in self.test_data_path.iterdir():
        name = str(file.absolute())
        if name.endswith('.json') or name.endswith('.repr'):
            seen.add(name[:-len('.json')])
        elif name.endswith('.json_inward') or name.endswith('.repr_inward'):
            seen.add(name[:-len('.json_inward')])
    return sorted(seen)
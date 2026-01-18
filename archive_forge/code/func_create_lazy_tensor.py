import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
def create_lazy_tensor(self, first_tensor_name: Optional[str]=None) -> str:
    if self.create_from_first_tensor:
        assert first_tensor_name is not None, 'Requires first tensor to create lazy tensor'
        return f'{first_tensor_name}.{self.create_tensor}'
    return f'{self.backend_namespace}::{self.create_tensor}'
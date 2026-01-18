import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@property
def buffers_to_mutate(self) -> Mapping[str, str]:
    return {s.arg.name: s.target for s in self.output_specs if s.kind == OutputKind.BUFFER_MUTATION and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}
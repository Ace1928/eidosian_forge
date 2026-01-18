import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, Tuple, Union
@property
def inputs_to_buffers(self) -> Mapping[str, str]:
    return {s.arg.name: s.target for s in self.input_specs if s.kind == InputKind.BUFFER and isinstance(s.arg, TensorArgument) and isinstance(s.target, str)}
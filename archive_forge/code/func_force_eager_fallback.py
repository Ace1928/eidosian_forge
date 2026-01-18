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
def force_eager_fallback(self, func: NativeFunction, schema: LazyIrSchema, metadata: BackendMetadata, sig: Union[DispatcherSignature, NativeSignature]) -> str:
    if self.gen_forced_fallback_code:
        return gen_fallback_code(schema, sig, overload_name=func.func.name.overload_name)
    return ''
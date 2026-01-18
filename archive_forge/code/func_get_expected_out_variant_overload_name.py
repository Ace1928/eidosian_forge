from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import Binding, DispatcherSignature, Expr
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import concatMap
def get_expected_out_variant_overload_name(overload_name: Optional[str]) -> str:
    return 'out' if not overload_name else f'{overload_name}_out'
import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
def gen_registration_helpers(backend_index: BackendIndex) -> List[str]:
    return [*gen_create_out_helper(backend_index), *gen_resize_out_helper(backend_index), *gen_check_inplace_helper(backend_index), *gen_maybe_create_proxy_helper(backend_index)]
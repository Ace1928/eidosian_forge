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
def gen_check_inplace_helper(backend_index: BackendIndex) -> List[str]:
    return ['\nvoid check_inplace(const Tensor &self, IntArrayRef sizes, const TensorOptions &options) {\n  // These checks are needed on those operators that:\n  //   1) don\'t use \'TensorIterator\' (e.g. \'addmm\' and \'baddbmm\')\n  //   2) have particular typing rules (e.g. \'cumsum\' and \'cumprod\')\n  // For other operators (e.g. \'add\'), \'TensorIterator\' already checks\n  // these things separately.\n  TORCH_CHECK(options.dtype() == self.dtype(),\n      "Bad in-place call: ",\n      "input tensor dtype ", self.dtype(), " and output tensor dtype ", options.dtype(), " should match");\n  TORCH_CHECK(options.device() == self.device(),\n      "Bad in-place call: ",\n      "input tensor device ", self.device(), " and output tensor device ", options.device(), " should match");\n  TORCH_CHECK(sizes == self.sizes(),\n      "Bad in-place call: ",\n      "input tensor size ", self.sizes(), " and output tensor size ", sizes, " should match");\n}\n']
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
def gen_class_set_output_functions(self, k: SchemaKind, parent_class: str, generate_super: bool) -> str:
    if generate_super:
        set_output_super = f'{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);'
    else:
        set_output_super = ''

    def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
        return f'\nvoid set_output_{name}(\n    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,\n    TensorOptions options, DimnameList names\n) override {{\n{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), '    ')}\n    if (!names.empty()) {{\n      namedinference::propagate_names(outputs_[output_idx], names);\n    }}\n    // super must happen after, so that downstream can use maybe_get_output\n    // to retrieve the output\n{textwrap.indent(set_output_super, '    ')}\n}}\n'
    return f'\n{gen_set_output_function('strided', maybe_create_proxy=True)}\n{gen_set_output_function('raw_strided', maybe_create_proxy=False)}\n'
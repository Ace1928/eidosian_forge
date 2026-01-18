import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
def out_variant(self, groups: Sequence[NativeFunctionsGroup]) -> str:
    if not groups:
        return ''
    generated_type_variants = []
    for g in groups:
        with native_function_manager(g):
            assert is_supported(g)
            assert isinstance(g, NativeFunctionsGroup)
            generated_type_variant = self.out_variant_op_test_case_generator(g)
            generated_type_variants.append(generated_type_variant)
    return '\n'.join(generated_type_variants)
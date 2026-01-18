import os
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Sequence, Set, Tuple
import numpy as np
from onnx import defs, helper
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets
from onnx.defs import ONNX_ML_DOMAIN, OpSchema
def generate_formal_parameter_tags(formal_parameter: OpSchema.FormalParameter) -> str:
    tags: List[str] = []
    if OpSchema.FormalParameterOption.Optional == formal_parameter.option:
        tags = ['optional']
    elif OpSchema.FormalParameterOption.Variadic == formal_parameter.option:
        if formal_parameter.is_homogeneous:
            tags = ['variadic']
        else:
            tags = ['variadic', 'heterogeneous']
    differentiable: OpSchema.DifferentiationCategory = OpSchema.DifferentiationCategory.Differentiable
    non_differentiable: OpSchema.DifferentiationCategory = OpSchema.DifferentiationCategory.NonDifferentiable
    if differentiable == formal_parameter.differentiation_category:
        tags.append('differentiable')
    elif non_differentiable == formal_parameter.differentiation_category:
        tags.append('non-differentiable')
    return '' if len(tags) == 0 else ' (' + ', '.join(tags) + ')'
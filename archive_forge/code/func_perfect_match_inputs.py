from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
@_beartype.beartype
def perfect_match_inputs(self, diagnostic: diagnostics.Diagnostic, args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], kwargs: Dict[str, fx_type_utils.Argument]) -> bool:
    """Check if the inputs perfectly match the OpSchema requirements.

        The definition of perfect match is that the input types are all in the type
        constraints and the number of inputs matches the number of inputs in the
        OpSchema.

        Checking steps:
        1. The function signature matches the inputs number, and attribute names.
        2. The input/attribute types are all in the type constraints.

        A function should at least pass the first step to be eligible for the
        nearest matching.

        Args:
            diagnostic: The diagnostic to use for logging detailed info.
            args: The input arguments organized in PyTorch inputs way.
            kwargs: The input keyword arguments organized in PyTorch inputs way.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
    function_inputs, function_attributes = self._separate_input_attributes_from_arguments(self.param_schema, args, kwargs, fill_defaults=True)
    with diagnostic.log_section(logging.INFO, 'Checking perfect match...'):
        diagnostic.info('%s', diagnostics.LazyString(diagnostics.format_argument, self.onnxfunction))
        is_perfect_match = True
        if len(function_inputs) != len(self.op_schema.inputs):
            with diagnostic.log_section(logging.INFO, 'Failed: input number mismatch!'):
                diagnostic.info('Actual %d vs expected %d', len(function_inputs), len(self.op_schema.inputs))
            diagnostic.info('The function is not a nearest match candidate.')
            is_perfect_match = False
        if set(function_attributes) != set(self.attributes):
            with diagnostic.log_section(logging.INFO, 'Failed: attribute mismatch!'):
                diagnostic.info('%s', diagnostics.LazyString(lambda: f'Actual {set(function_attributes)} vs expected {set(self.attributes)}'))
            diagnostic.info('The function is not a nearest match candidate.')
            is_perfect_match = False
        if not is_perfect_match:
            return False
        for schema_input, torch_input in zip(self.op_schema.inputs, function_inputs):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if not allowed_types.intersection(torch_input_compatible_types) and (not any((fx_type_utils.is_optional_onnx_dtype_str(onnx_type_str) for onnx_type_str in allowed_types))):
                with diagnostic.log_section(logging.INFO, "Failed: input type mismatch for input '%s'!", schema_input.name):
                    diagnostic.info('Actual %s vs\nExpected %s', torch_input_compatible_types, allowed_types)
                is_perfect_match = False
        for attribute_name, attribute in function_attributes.items():
            if not self._match_onnx_attribute_type(attribute_name, attribute):
                with diagnostic.log_section(logging.INFO, "Failed: attribute '%s' type mismatch!", attribute_name):
                    diagnostic.info('Actual %s vs\nExpected %s', type(attribute), self.attributes[attribute_name].type)
                is_perfect_match = False
        self._record_matching_score(function_inputs, function_attributes)
        diagnostic.info('match score: %d', self.match_score)
        return is_perfect_match
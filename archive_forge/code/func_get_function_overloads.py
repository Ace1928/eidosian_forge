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
@diagnostics.diagnose_call(diagnostics.rules.find_operator_overloads_in_onnx_registry, diagnostic_message_formatter=_find_operator_overloads_in_onnx_registry_disagnostic_message_formatter)
def get_function_overloads(self, node: torch.fx.Node, diagnostic_context: diagnostics.DiagnosticContext) -> List[registration.ONNXFunction]:
    """Get the function overloads from the registry.

        Args:
            node: The node to get the function overloads for.
            diagnostic_context: The diagnostic context to use for reporting errors.

        Returns:
            The list contains ONNXFunctions, starting with the default ones and
            followed by any custom ones.
        """
    internal_opname: registration.OpName = self._get_aten_name(node=node, diagnostic_context=diagnostic_context)
    function_group: Optional[List[registration.ONNXFunction]] = None
    function_group = self.onnx_registry.get_op_functions(namespace=internal_opname.namespace, op_name=internal_opname.op_name, overload=internal_opname.overload)
    if function_group is None:
        function_group = self.onnx_registry.get_op_functions(namespace=internal_opname.namespace, op_name=internal_opname.op_name, overload=None)
        if function_group is not None:
            op_full_name = internal_opname.qualified_name()
            diagnostic = diagnostic_context.inflight_diagnostic()
            diagnostic.warning('### The operator overload is not found in onnx registry!\nCannot find the operator overload in onnx registry, but the default overload is found. Please check the ONNX output carefully. \n')
            diagnostic.level = diagnostics.levels.WARNING
    if function_group is not None:
        function_group = self._filter_or_keep_complex(node, function_group, diagnostic_context)
        return function_group
    op_full_name = internal_opname.qualified_name()
    diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find symbolic function for {op_full_name}, which should be registered under {node.target}.', unsupported_fx_node=node)
    diagnostic_context.log(diagnostic)
    raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
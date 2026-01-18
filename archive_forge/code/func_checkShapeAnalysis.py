import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from itertools import chain
from typing import List, Union
from torch._C import TensorType
import io
def checkShapeAnalysis(self, out_sizes: Union[List[int], List[List[int]]], traced_graph, assert_propagation, constant_prop=True):
    prev_symbolic_shapes_test_enabled = torch._C._jit_symbolic_shapes_test_mode_enabled()
    for enable_test_mode in [True, False]:
        torch._C._jit_set_symbolic_shapes_test_mode(enable_test_mode)
        torch._C._jit_erase_non_input_shape_information(traced_graph)
        if constant_prop:
            torch._C._jit_pass_constant_propagation(traced_graph)
        torch._C._jit_pass_propagate_shapes_on_graph(traced_graph)
        output = next(traced_graph.outputs()).type()

        def test_type(type, actual_size):
            sizes = type.symbolic_sizes()
            out_type = TensorType.get().with_sizes(sizes)
            actual_type = TensorType.get().with_sizes(actual_size)
            self.assertTrue(actual_type.isSubtypeOf(out_type))
            if assert_propagation:
                self.assertEqual(out_type.sizes(), actual_size)
        if output.isSubtypeOf(torch._C.TensorType.get()):
            test_type(output, out_sizes)
        else:
            tuple_elements = output.elements()
            for i in range(len(tuple_elements)):
                test_type(tuple_elements[i], out_sizes[i])
    torch._C._jit_set_symbolic_shapes_test_mode(prev_symbolic_shapes_test_enabled)
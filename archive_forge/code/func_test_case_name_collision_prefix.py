import unittest
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np
from onnx import (
def test_case_name_collision_prefix(self) -> None:
    """Tests a scenario where we merge two models that have name collisions, but they
        are avoided by prefixing the models model.
        """
    m1_def = '\n            <\n                ir_version: 7,\n                opset_import: [ "": 10]\n            >\n            agraph (float[N] A, float[N] B) => (float[N] C)\n            {\n                C = Add(A, B)\n            }\n            '
    io_map = [('C', 'A')]

    def check_expectations(g1: GraphProto, g2: GraphProto, g3: GraphProto) -> None:
        del g1, g2
        self.assertEqual(['m1/A', 'm1/B', 'm2/B'], [elem.name for elem in g3.input])
        self.assertEqual(['m2/C'], [elem.name for elem in g3.output])
        self.assertEqual(['Add', 'Add'], [elem.op_type for elem in g3.node])
    self._test_merge_models(m1_def, m1_def, io_map, check_expectations, prefix1='m1/', prefix2='m2/')
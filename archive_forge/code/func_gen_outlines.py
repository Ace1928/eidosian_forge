import os
from typing import IO, Any, Dict, List, Sequence
from onnx import AttributeProto, defs, load
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner import Runner
def gen_outlines(f: IO[Any], ml: bool) -> None:
    f.write('# Test Coverage Report')
    if ml:
        f.write(' (ONNX-ML Operators)\n')
    else:
        f.write(' (ONNX Core Operators)\n')
    f.write('## Outlines\n')
    f.write('* [Node Test Coverage](#node-test-coverage)\n')
    f.write('* [Model Test Coverage](#model-test-coverage)\n')
    f.write('* [Overall Test Coverage](#overall-test-coverage)\n')
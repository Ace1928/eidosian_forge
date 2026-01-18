import os
from typing import IO, Any, Dict, List, Sequence
from onnx import AttributeProto, defs, load
from onnx.backend.test.case import collect_snippets
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner import Runner
def is_ml(schemas: Sequence[defs.OpSchema]) -> bool:
    return any((s.domain == 'ai.onnx.ml' for s in schemas))
from typing import List
import onnx.onnx_cpp2py_export.defs as C  # noqa: N812
from onnx import AttributeProto, FunctionProto
def onnx_ml_opset_version() -> int:
    """Return current opset for domain `ai.onnx.ml`."""
    return C.schema_version_map()[ONNX_ML_DOMAIN][1]
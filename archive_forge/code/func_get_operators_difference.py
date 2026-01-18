import gc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import onnx
from onnx import load_model
from transformers.models.auto.configuration_auto import AutoConfig
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.transformers.optimizer import optimize_model
from ..onnx.utils import check_model_uses_external_data
from ..utils import CONFIG_NAME, NormalizedConfigManager, logging
from ..utils.save_utils import maybe_save_preprocessors
from .configuration import OptimizationConfig, ORTConfig
from .modeling_decoder import ORTModelForCausalLM
from .modeling_ort import ORTModel
from .modeling_seq2seq import ORTModelForConditionalGeneration
from .utils import ONNX_WEIGHTS_NAME, ORTConfigManager
@staticmethod
def get_operators_difference(onnx_model_path: Union[str, os.PathLike], onnx_optimized_model_path: Union[str, os.PathLike]) -> Dict[str, int]:
    """
        Compute the dictionary mapping the operators name to the difference in the number of corresponding nodes between
        the original and the optimized model.

        Args:
            onnx_model_path (`Union[str, os.PathLike]`):
                Path of the ONNX model.
            onnx_optimized_model_path (`Union[str, os.PathLike]`):
                Path of the optimized ONNX model.

        Returns:
            The dictionary mapping the operators name to the difference in the number of corresponding nodes between the
            original and the optimized model.
        """
    onnx_model = BertOnnxModel(load_model(onnx_model_path))
    onnx_optimized_model = BertOnnxModel(load_model(onnx_optimized_model_path))

    def nodes_difference_given_type(op_type):
        onnx_model_nodes_with_op_type = len(onnx_model.get_nodes_by_op_type(op_type))
        onnx_optimized_model_nodes_with_op_type = len(onnx_optimized_model.get_nodes_by_op_type(op_type))
        return onnx_model_nodes_with_op_type - onnx_optimized_model_nodes_with_op_type
    op_types = set()
    for model in [onnx_model, onnx_optimized_model]:
        for node in model.nodes():
            op_types.add(node.op_type)
    operators_difference = {op_type: nodes_difference_given_type(op_type) for op_type in op_types}
    return {k: v for k, v in operators_difference.items() if v != 0}
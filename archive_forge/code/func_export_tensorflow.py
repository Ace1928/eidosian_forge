import warnings
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union
import numpy as np
from packaging.version import Version, parse
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import (
from .config import OnnxConfig
def export_tensorflow(preprocessor: Union['PreTrainedTokenizer', 'FeatureExtractionMixin'], model: 'TFPreTrainedModel', config: OnnxConfig, opset: int, output: Path, tokenizer: 'PreTrainedTokenizer'=None) -> Tuple[List[str], List[str]]:
    """
    Export a TensorFlow model to an ONNX Intermediate Representation (IR)

    Args:
        preprocessor: ([`PreTrainedTokenizer`] or [`FeatureExtractionMixin`]):
            The preprocessor used for encoding the data.
        model ([`TFPreTrainedModel`]):
            The model to export.
        config ([`~onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    import onnx
    import tensorflow as tf
    import tf2onnx
    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError('You cannot provide both a tokenizer and preprocessor to export the model.')
    if tokenizer is not None:
        warnings.warn('The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use `preprocessor` instead.', FutureWarning)
        logger.info('Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.')
        preprocessor = tokenizer
    model.config.return_dict = True
    if config.values_override is not None:
        logger.info(f'Overriding {len(config.values_override)} configuration item(s)')
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f'\t- {override_config_key} -> {override_config_value}')
            setattr(model.config, override_config_key, override_config_value)
    model_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.TENSORFLOW)
    inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
    onnx_outputs = list(config.outputs.keys())
    input_signature = [tf.TensorSpec([None] * tensor.ndim, dtype=tensor.dtype, name=key) for key, tensor in model_inputs.items()]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=opset)
    onnx.save(onnx_model, output.as_posix())
    config.restore_ops()
    return (matched_inputs, onnx_outputs)
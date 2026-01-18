import warnings
from argparse import ArgumentParser
from os import listdir, makedirs
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from packaging.version import Version, parse
from transformers.pipelines import Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import ModelOutput, is_tf_available, is_torch_available
def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR)

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model

    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow

    """
    if not is_tf_available():
        raise Exception('Cannot convert because TF is not installed. Please install tensorflow first.')
    print("/!\\ Please note TensorFlow doesn't support exporting model > 2Gb /!\\")
    try:
        import tensorflow as tf
        import tf2onnx
        from tf2onnx import __version__ as t2ov
        print(f'Using framework TensorFlow: {tf.version.VERSION}, tf2onnx: {t2ov}')
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, 'tf')
        nlp.model.predict(tokens.data)
        input_signature = [tf.TensorSpec.from_tensor(tensor, name=key) for key, tensor in tokens.items()]
        model_proto, _ = tf2onnx.convert.from_keras(nlp.model, input_signature, opset=opset, output_path=output.as_posix())
    except ImportError as e:
        raise Exception(f'Cannot import {e.name} required to convert TF model to ONNX. Please install {e.name} first. {e}')
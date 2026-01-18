import os
from functools import partial, reduce
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Type, Union
import transformers
from .. import PretrainedConfig, is_tf_available, is_torch_available
from ..utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from .config import OnnxConfig
@staticmethod
def determine_framework(model: str, framework: str=None) -> str:
    """
        Determines the framework to use for the export.

        The priority is in the following order:
            1. User input via `framework`.
            2. If local checkpoint is provided, use the same framework as the checkpoint.
            3. Available framework in environment, with priority given to PyTorch

        Args:
            model (`str`):
                The name of the model to export.
            framework (`str`, *optional*, defaults to `None`):
                The framework to use for the export. See above for priority if none provided.

        Returns:
            The framework to use for the export.

        """
    if framework is not None:
        return framework
    framework_map = {'pt': 'PyTorch', 'tf': 'TensorFlow'}
    exporter_map = {'pt': 'torch', 'tf': 'tf2onnx'}
    if os.path.isdir(model):
        if os.path.isfile(os.path.join(model, WEIGHTS_NAME)):
            framework = 'pt'
        elif os.path.isfile(os.path.join(model, TF2_WEIGHTS_NAME)):
            framework = 'tf'
        else:
            raise FileNotFoundError(f'Cannot determine framework from given checkpoint location. There should be a {WEIGHTS_NAME} for PyTorch or {TF2_WEIGHTS_NAME} for TensorFlow.')
        logger.info(f'Local {framework_map[framework]} model found.')
    elif is_torch_available():
        framework = 'pt'
    elif is_tf_available():
        framework = 'tf'
    else:
        raise EnvironmentError('Neither PyTorch nor TensorFlow found in environment. Cannot export to ONNX.')
    logger.info(f'Framework not requested. Using {exporter_map[framework]} to export to ONNX.')
    return framework
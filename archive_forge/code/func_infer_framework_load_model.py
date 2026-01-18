import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
def infer_framework_load_model(model, config: AutoConfig, model_classes: Optional[Dict[str, Tuple[type]]]=None, task: Optional[str]=None, framework: Optional[str]=None, **model_kwargs):
    """
    Select framework (TensorFlow or PyTorch) to use from the `model` passed. Returns a tuple (framework, model).

    If `model` is instantiated, this function will just infer the framework from the model class. Otherwise `model` is
    actually a checkpoint name and this method will try to instantiate it using `model_classes`. Since we don't want to
    instantiate the model twice, this model is returned for use by the pipeline.

    If both frameworks are installed and available for `model`, PyTorch is selected.

    Args:
        model (`str`, [`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to infer the framework from. If `str`, a checkpoint name. The model to infer the framewrok from.
        config ([`AutoConfig`]):
            The config associated with the model to help using the correct class
        model_classes (dictionary `str` to `type`, *optional*):
            A mapping framework to class.
        task (`str`):
            The task defining which pipeline will be returned.
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
            **model_kwargs)` function.

    Returns:
        `Tuple`: A tuple framework, model.
    """
    if not is_tf_available() and (not is_torch_available()):
        raise RuntimeError('At least one of TensorFlow 2.0 or PyTorch should be installed. To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ To install PyTorch, read the instructions at https://pytorch.org/.')
    if isinstance(model, str):
        model_kwargs['_from_pipeline'] = task
        class_tuple = ()
        look_pt = is_torch_available() and framework in {'pt', None}
        look_tf = is_tf_available() and framework in {'tf', None}
        if model_classes:
            if look_pt:
                class_tuple = class_tuple + model_classes.get('pt', (AutoModel,))
            if look_tf:
                class_tuple = class_tuple + model_classes.get('tf', (TFAutoModel,))
        if config.architectures:
            classes = []
            for architecture in config.architectures:
                transformers_module = importlib.import_module('transformers')
                if look_pt:
                    _class = getattr(transformers_module, architecture, None)
                    if _class is not None:
                        classes.append(_class)
                if look_tf:
                    _class = getattr(transformers_module, f'TF{architecture}', None)
                    if _class is not None:
                        classes.append(_class)
            class_tuple = class_tuple + tuple(classes)
        if len(class_tuple) == 0:
            raise ValueError(f'Pipeline cannot infer suitable model classes from {model}')
        all_traceback = {}
        for model_class in class_tuple:
            kwargs = model_kwargs.copy()
            if framework == 'pt' and model.endswith('.h5'):
                kwargs['from_tf'] = True
                logger.warning('Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. Trying to load the model with PyTorch.')
            elif framework == 'tf' and model.endswith('.bin'):
                kwargs['from_pt'] = True
                logger.warning('Model might be a PyTorch model (ending with `.bin`) but PyTorch is not available. Trying to load the model with Tensorflow.')
            try:
                model = model_class.from_pretrained(model, **kwargs)
                if hasattr(model, 'eval'):
                    model = model.eval()
                break
            except (OSError, ValueError):
                all_traceback[model_class.__name__] = traceback.format_exc()
                continue
        if isinstance(model, str):
            error = ''
            for class_name, trace in all_traceback.items():
                error += f'while loading with {class_name}, an error is thrown:\n{trace}\n'
            raise ValueError(f'Could not load model {model} with any of the following classes: {class_tuple}. See the original errors:\n\n{error}\n')
    if framework is None:
        framework = infer_framework(model.__class__)
    return (framework, model)
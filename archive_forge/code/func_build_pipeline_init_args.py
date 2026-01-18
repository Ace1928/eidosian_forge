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
def build_pipeline_init_args(has_tokenizer: bool=False, has_feature_extractor: bool=False, has_image_processor: bool=False, supports_binary_output: bool=True) -> str:
    docstring = '\n    Arguments:\n        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):\n            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from\n            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.'
    if has_tokenizer:
        docstring += '\n        tokenizer ([`PreTrainedTokenizer`]):\n            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from\n            [`PreTrainedTokenizer`].'
    if has_feature_extractor:
        docstring += '\n        feature_extractor ([`SequenceFeatureExtractor`]):\n            The feature extractor that will be used by the pipeline to encode data for the model. This object inherits from\n            [`SequenceFeatureExtractor`].'
    if has_image_processor:
        docstring += '\n        image_processor ([`BaseImageProcessor`]):\n            The image processor that will be used by the pipeline to encode data for the model. This object inherits from\n            [`BaseImageProcessor`].'
    docstring += '\n        modelcard (`str` or [`ModelCard`], *optional*):\n            Model card attributed to the model for this pipeline.\n        framework (`str`, *optional*):\n            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be\n            installed.\n\n            If no framework is specified, will default to the one currently installed. If no framework is specified and\n            both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is\n            provided.\n        task (`str`, defaults to `""`):\n            A task-identifier for the pipeline.\n        num_workers (`int`, *optional*, defaults to 8):\n            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the number of\n            workers to be used.\n        batch_size (`int`, *optional*, defaults to 1):\n            When the pipeline will use *DataLoader* (when passing a dataset, on GPU for a Pytorch model), the size of\n            the batch to use, for inference this is not always beneficial, please read [Batching with\n            pipelines](https://huggingface.co/transformers/main_classes/pipelines.html#pipeline-batching) .\n        args_parser ([`~pipelines.ArgumentHandler`], *optional*):\n            Reference to the object in charge of parsing supplied pipeline parameters.\n        device (`int`, *optional*, defaults to -1):\n            Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run the model on\n            the associated CUDA device id. You can pass native `torch.device` or a `str` too\n        torch_dtype (`str` or `torch.dtype`, *optional*):\n            Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model\n            (`torch.float16`, `torch.bfloat16`, ... or `"auto"`)'
    if supports_binary_output:
        docstring += '\n        binary_output (`bool`, *optional*, defaults to `False`):\n            Flag indicating if the output the pipeline should happen in a serialized format (i.e., pickle) or as\n            the raw output data e.g. text.'
    return docstring
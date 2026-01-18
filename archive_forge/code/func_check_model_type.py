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
def check_model_type(self, supported_models: Union[List[str], dict]):
    """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (`List[str]` or `dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
    if not isinstance(supported_models, list):
        supported_models_names = []
        for _, model_name in supported_models.items():
            if isinstance(model_name, tuple):
                supported_models_names.extend(list(model_name))
            else:
                supported_models_names.append(model_name)
        if hasattr(supported_models, '_model_mapping'):
            for _, model in supported_models._model_mapping._extra_content.items():
                if isinstance(model_name, tuple):
                    supported_models_names.extend([m.__name__ for m in model])
                else:
                    supported_models_names.append(model.__name__)
        supported_models = supported_models_names
    if self.model.__class__.__name__ not in supported_models:
        logger.error(f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are {supported_models}.")
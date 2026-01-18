import contextlib
import importlib
import json
import logging
import os
import re
import shutil
import types
import warnings
from functools import lru_cache
from importlib.util import find_spec
from typing import Callable, NamedTuple
import cloudpickle
import yaml
from packaging import version
from packaging.version import Version
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
from mlflow.utils.class_utils import _get_class_from_string
def register_pydantic_serializer():
    """
    Helper function to pickle pydantic fields for pydantic v1.
    Pydantic's Cython validators are not serializable.
    https://github.com/cloudpipe/cloudpickle/issues/408
    """
    import pydantic
    if Version(pydantic.__version__) >= Version('2.0.0'):
        return
    import pydantic.fields

    def custom_serializer(obj):
        return {'name': obj.name, 'type_': obj.outer_type_, 'class_validators': obj.class_validators, 'model_config': obj.model_config, 'default': obj.default, 'default_factory': obj.default_factory, 'required': obj.required, 'final': obj.final, 'alias': obj.alias, 'field_info': obj.field_info}

    def custom_deserializer(kwargs):
        return pydantic.fields.ModelField(**kwargs)

    def _CloudPicklerReducer(obj):
        return (custom_deserializer, (custom_serializer(obj),))
    warnings.warn('Using custom serializer to pickle pydantic.fields.ModelField classes, this might miss some fields and validators. To avoid this, please upgrade pydantic to v2 using `pip install pydantic -U` with langchain 0.0.267 and above.')
    cloudpickle.CloudPickler.dispatch[pydantic.fields.ModelField] = _CloudPicklerReducer
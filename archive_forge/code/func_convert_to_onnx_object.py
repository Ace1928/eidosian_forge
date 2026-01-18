from . import _catboost
from .core import Pool, CatBoostError, ARRAY_TYPES, PATH_TYPES, fspath, _update_params_quantize_part, _process_synonyms
from collections import defaultdict
from contextlib import contextmanager
import sys
import numpy as np
import warnings
def convert_to_onnx_object(model, export_parameters=None, **kwargs):
    """
    Convert given CatBoost model to ONNX-ML model.
    Categorical Features are not supported.

    Parameters
    ----------
    model : CatBoost trained model
    export_parameters : dict [default=None]
        Parameters for ONNX-ML export:
            * onnx_graph_name : string
                The name property of onnx Graph
            * onnx_domain : string
                The domain component of onnx Model
            * onnx_model_version : int
                The model_version component of onnx Model
            * onnx_doc_string : string
                The doc_string component of onnx Model
    Returns
    -------
    onnx_object : ModelProto
        The model in ONNX format
    """
    try:
        import onnx
    except ImportError as e:
        warnings.warn('To get working onnx model you should install onnx.')
        raise ImportError(str(e))
    import json
    if not model.is_fitted():
        raise CatBoostError('There is no trained model to use save_model(). Use fit() to train model. Then use this method.')
    for name, value in kwargs.items():
        if name == 'target_opset' and value not in [None, 2]:
            warnings.warn('target_opset argument is not supported. Default target_opset is 2 (ai.onnx.ml domain)')
        elif name == 'initial_types' and value is not None:
            warnings.warn('initial_types argument is not supported')
    params_string = ''
    if export_parameters:
        params_string = json.dumps(export_parameters, cls=_NumpyAwareEncoder)
    model_str = _get_onnx_model(model._object, params_string)
    onnx_model = onnx.load_model_from_string(model_str)
    return onnx_model
import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
def load_tf_model(model_path, tags=(SERVING,), config=None):
    """Loads the model at the specified path.

  Args:
    model_path: the path to either session_bundle or SavedModel
    tags: the tags that determines the model to load.
    config: tf.ConfigProto containing session configuration options.

  Returns:
    A pair of (Session, map<string, SignatureDef>) objects.

  Raises:
    PredictionError: if the model could not be loaded.
  """
    _load_tf_custom_op(model_path)
    if tf.saved_model.loader.maybe_saved_model_directory(model_path):
        try:
            logging.info('Importing tensorflow.contrib in load_tf_model')
            if tf.__version__.startswith('1.0'):
                session = tf.Session(target='', graph=None, config=config)
            else:
                session = tf.Session(target='', graph=tf.Graph(), config=config)
            meta_graph = tf.saved_model.loader.load(session, tags=list(tags), export_dir=model_path)
        except Exception as e:
            msg = 'Failed to load the model due to bad model data. tags: %s' % (list(tags),)
            logging.exception(msg)
            raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, '%s\n%s' % (msg, str(e)))
    else:
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, 'Cloud ML only supports TF 1.0 or above and models saved in SavedModel format.')
    if session is None:
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, 'Failed to create session when loading the model')
    if not meta_graph.signature_def:
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, 'MetaGraph must have at least one signature_def.')
    invalid_signatures = []
    for signature_name in meta_graph.signature_def:
        try:
            signature = meta_graph.signature_def[signature_name]
            _update_dtypes(session.graph, signature.inputs)
            _update_dtypes(session.graph, signature.outputs)
        except ValueError as e:
            logging.warn('Error updating signature %s: %s', signature_name, str(e))
            invalid_signatures.append(signature_name)
    for signature_name in invalid_signatures:
        del meta_graph.signature_def[signature_name]
    return (session, meta_graph.signature_def)
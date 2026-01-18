from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def after_create_session(self, session, coord):
    """Does first run which shows the eval metrics before training."""
    if tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SAVEABLE_OBJECTS):
        raise ValueError('InMemoryEvaluator does not support saveables other than global variables.')
    self._var_name_to_train_var = {v.name: v for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)}
    var_names_to_transfer = set(self._var_name_to_placeholder.keys()) & set(self._var_name_to_train_var.keys())
    self._var_name_to_train_var = {v_name: self._var_name_to_train_var[v_name] for v_name in var_names_to_transfer}
    self._var_name_to_eval_var = {v_name: self._var_name_to_eval_var[v_name] for v_name in var_names_to_transfer}
    with self._graph.as_default():
        self._var_feed_op = tf.group([tf.compat.v1.assign(self._var_name_to_eval_var[v_name], self._var_name_to_placeholder[v_name]) for v_name in var_names_to_transfer])
    self._evaluate(session)
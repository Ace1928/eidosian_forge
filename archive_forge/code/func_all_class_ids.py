from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _NumericColumn
from tensorflow.python.framework import ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def all_class_ids(logits, n_classes):
    batch_size = tf.compat.v1.shape(logits)[0]
    class_id_list = tf.range(n_classes)
    return tf.tile(input=tf.compat.v1.expand_dims(input=class_id_list, axis=0), multiples=[batch_size, 1])
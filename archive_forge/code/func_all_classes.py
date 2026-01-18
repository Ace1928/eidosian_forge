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
def all_classes(logits, n_classes, label_vocabulary=None):
    batch_size = tf.compat.v1.shape(logits)[0]
    if label_vocabulary:
        classes_list = tf.convert_to_tensor([label_vocabulary])
    else:
        classes_list = tf.expand_dims(tf.range(n_classes), axis=0)
        classes_list = tf.strings.as_string(classes_list)
    return tf.tile(input=classes_list, multiples=[batch_size, 1])
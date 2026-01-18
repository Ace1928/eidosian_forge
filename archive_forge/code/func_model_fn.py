from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clustering_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
def model_fn(self, features, mode, config):
    """Model function for the estimator.

    Note that this does not take a `labels` arg. This works, but `input_fn` must
    return either `features` or, equivalently, `(features, None)`.

    Args:
      features: The input points. See `tf.estimator.Estimator`.
      mode: See `tf.estimator.Estimator`.
      config: See `tf.estimator.Estimator`.

    Returns:
      A `tf.estimator.EstimatorSpec` (see `tf.estimator.Estimator`) specifying
      this behavior:
        * `train_op`: Execute one mini-batch or full-batch run of Lloyd's
             algorithm.
        * `loss`: The sum of the squared distances from each input point to its
             closest center.
        * `eval_metric_ops`: Maps `SCORE` to `loss`.
        * `predictions`: Maps `ALL_DISTANCES` to the distance from each input
             point to each cluster center; maps `CLUSTER_INDEX` to the index of
             the closest cluster center for each input point.
    """
    input_points = _parse_features_if_necessary(features, self._feature_columns)
    all_distances, model_predictions, losses, is_initialized, init_op, training_op = clustering_ops.KMeans(inputs=input_points, num_clusters=self._num_clusters, initial_clusters=self._initial_clusters, distance_metric=self._distance_metric, use_mini_batch=self._use_mini_batch, mini_batch_steps_per_iteration=self._mini_batch_steps_per_iteration, random_seed=self._seed, kmeans_plus_plus_num_retries=self._kmeans_plus_plus_num_retries).training_graph()
    loss = tf.math.reduce_sum(losses)
    tf.compat.v1.summary.scalar('loss/raw', loss)
    incr_step = tf.compat.v1.assign_add(tf.compat.v1.train.get_global_step(), 1)
    training_op = control_flow_ops.with_dependencies([training_op, incr_step], loss)
    training_hooks = [_InitializeClustersHook(init_op, is_initialized, config.is_chief)]
    if self._relative_tolerance is not None:
        training_hooks.append(_LossRelativeChangeHook(loss, self._relative_tolerance))
    export_outputs = {KMeansClustering.ALL_DISTANCES: export_output.PredictOutput(all_distances[0]), KMeansClustering.CLUSTER_INDEX: export_output.PredictOutput(model_predictions[0]), tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: export_output.PredictOutput(model_predictions[0])}
    return model_fn_lib.EstimatorSpec(mode=mode, predictions={KMeansClustering.ALL_DISTANCES: all_distances[0], KMeansClustering.CLUSTER_INDEX: model_predictions[0]}, loss=loss, train_op=training_op, eval_metric_ops={KMeansClustering.SCORE: tf.compat.v1.metrics.mean(loss)}, training_hooks=training_hooks, export_outputs=export_outputs)
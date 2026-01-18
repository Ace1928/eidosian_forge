from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import ops
from tensorflow.python.tpu import feature_column as tpu_fc
from tensorflow.python.tpu import feature_column_v2 as tpu_fc_v2
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu.tpu_embedding import AdagradParameters
from tensorflow.python.tpu.tpu_embedding import AdamParameters
from tensorflow.python.tpu.tpu_embedding import FtrlParameters
from tensorflow.python.tpu.tpu_embedding import MomentumParameters
from tensorflow.python.tpu.tpu_embedding import ProximalAdagradParameters
from tensorflow.python.tpu.tpu_embedding import RMSPropParameters
from tensorflow.python.tpu.tpu_embedding import StochasticGradientDescentParameters
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def split_inputs(ctx, features, labels, num_cores_per_batch=1):
    """Splits the dense and sparse tensors inside the features and labels."""
    enqueue_datas = collections.OrderedDict()
    if ctx.embedding_config:
        tpu_embedding_ = ctx.embedding_config.tpu_embedding
        for feature_key in tpu_embedding_.feature_to_config_dict:
            sparse_feature = _get_sparse_feature_from_feature(feature_key, features)
            max_sequence_length = tpu_embedding_.feature_to_config_dict[feature_key].max_sequence_length
            combiner = tpu_embedding_._table_to_config_dict[tpu_embedding_._feature_to_config_dict[feature_key].table_id].combiner
            if max_sequence_length > 0:
                length_feature_name = tpu_fc.get_sequence_length_feature_key_name_from_feature_key_name(feature_key)
                length_feature = tf.math.minimum(fc_utils.sequence_length_from_sparse_tensor(sparse_feature), max_sequence_length)
                length_feature.set_shape(ctx.batch_size_for_input_fn)
                features[length_feature_name] = length_feature
            weight_key = tpu_embedding_.feature_to_config_dict[feature_key].weight_key
            sparse_feature_split = _split_tensor(sparse_feature, num_cores_per_batch)
            if combiner is None and (not isinstance(sparse_feature, tf.sparse.SparseTensor)):
                if weight_key is not None:
                    raise ValueError('Found weights {} for weighted_categorical_column, which is notcompatible with sparse feature {} enqueued as dense tensor.'.format(weight_key, feature_key))
                enqueue_data = []
                for i in range(num_cores_per_batch):
                    enqueue_data.append(tpu_embedding.EnqueueData(sparse_feature_split[i]))
            else:
                weights = None
                if isinstance(sparse_feature, tf.sparse.SparseTensor):
                    weights = _get_weights_from_features(weight_key, features)
                    weights_split = _split_tensor(weights, num_cores_per_batch)
                enqueue_data = []
                for i in range(num_cores_per_batch):
                    split_weights = weights_split[i] if weights else None
                    enqueue_data.append(tpu_embedding.EnqueueData.from_sparse_tensor(_maybe_dense_to_sparse(sparse_feature_split[i]), weights=split_weights))
            enqueue_datas[feature_key] = enqueue_data
    if ctx.tensor_core_embedding_columns:
        for column in ctx.tensor_core_embedding_columns:
            feature_key = column.categorical_column.key
            sparse_feature = _get_sparse_feature_from_feature(feature_key, features)
            padded_values, padded_mask = tpu_fc_v2.pad_sparse_embedding_lookup_indices(sparse_feature, column._tensor_core_shape[1])
            padded_values.set_shape([ctx.batch_size_for_input_fn, column._tensor_core_shape[1]])
            padded_mask.set_shape([ctx.batch_size_for_input_fn, column._tensor_core_shape[1]])
            features[feature_key] = padded_values
            mask_key = feature_key + tpu_fc_v2._TENSOR_CORE_MASK_KEY_SUFFIX
            if mask_key in features:
                raise ValueError('Mask key {} for Tensor Core embedding is already in use.'.format(mask_key))
            features[mask_key] = padded_mask
    enqueue_datas_list = []
    for i in range(num_cores_per_batch):
        enqueue_data = {}
        for key, value in enqueue_datas.items():
            enqueue_data[key] = value[i]
        enqueue_datas_list.append(enqueue_data)
    return (features, labels, enqueue_datas_list)
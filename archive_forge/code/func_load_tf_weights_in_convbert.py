import math
import os
from operator import attrgetter
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, get_activation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_convbert import ConvBertConfig
def load_tf_weights_in_convbert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import tensorflow as tf
    except ImportError:
        logger.error('Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.')
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    init_vars = tf.train.list_variables(tf_path)
    tf_data = {}
    for name, shape in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_path, name)
        tf_data[name] = array
    param_mapping = {'embeddings.word_embeddings.weight': 'electra/embeddings/word_embeddings', 'embeddings.position_embeddings.weight': 'electra/embeddings/position_embeddings', 'embeddings.token_type_embeddings.weight': 'electra/embeddings/token_type_embeddings', 'embeddings.LayerNorm.weight': 'electra/embeddings/LayerNorm/gamma', 'embeddings.LayerNorm.bias': 'electra/embeddings/LayerNorm/beta', 'embeddings_project.weight': 'electra/embeddings_project/kernel', 'embeddings_project.bias': 'electra/embeddings_project/bias'}
    if config.num_groups > 1:
        group_dense_name = 'g_dense'
    else:
        group_dense_name = 'dense'
    for j in range(config.num_hidden_layers):
        param_mapping[f'encoder.layer.{j}.attention.self.query.weight'] = f'electra/encoder/layer_{j}/attention/self/query/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.query.bias'] = f'electra/encoder/layer_{j}/attention/self/query/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.key.weight'] = f'electra/encoder/layer_{j}/attention/self/key/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.key.bias'] = f'electra/encoder/layer_{j}/attention/self/key/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.value.weight'] = f'electra/encoder/layer_{j}/attention/self/value/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.value.bias'] = f'electra/encoder/layer_{j}/attention/self/value/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.key_conv_attn_layer.depthwise.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_key/depthwise_kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.key_conv_attn_layer.pointwise.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_key/pointwise_kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.key_conv_attn_layer.bias'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_key/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_kernel_layer.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_kernel/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_kernel_layer.bias'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_kernel/bias'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_out_layer.weight'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_point/kernel'
        param_mapping[f'encoder.layer.{j}.attention.self.conv_out_layer.bias'] = f'electra/encoder/layer_{j}/attention/self/conv_attn_point/bias'
        param_mapping[f'encoder.layer.{j}.attention.output.dense.weight'] = f'electra/encoder/layer_{j}/attention/output/dense/kernel'
        param_mapping[f'encoder.layer.{j}.attention.output.LayerNorm.weight'] = f'electra/encoder/layer_{j}/attention/output/LayerNorm/gamma'
        param_mapping[f'encoder.layer.{j}.attention.output.dense.bias'] = f'electra/encoder/layer_{j}/attention/output/dense/bias'
        param_mapping[f'encoder.layer.{j}.attention.output.LayerNorm.bias'] = f'electra/encoder/layer_{j}/attention/output/LayerNorm/beta'
        param_mapping[f'encoder.layer.{j}.intermediate.dense.weight'] = f'electra/encoder/layer_{j}/intermediate/{group_dense_name}/kernel'
        param_mapping[f'encoder.layer.{j}.intermediate.dense.bias'] = f'electra/encoder/layer_{j}/intermediate/{group_dense_name}/bias'
        param_mapping[f'encoder.layer.{j}.output.dense.weight'] = f'electra/encoder/layer_{j}/output/{group_dense_name}/kernel'
        param_mapping[f'encoder.layer.{j}.output.dense.bias'] = f'electra/encoder/layer_{j}/output/{group_dense_name}/bias'
        param_mapping[f'encoder.layer.{j}.output.LayerNorm.weight'] = f'electra/encoder/layer_{j}/output/LayerNorm/gamma'
        param_mapping[f'encoder.layer.{j}.output.LayerNorm.bias'] = f'electra/encoder/layer_{j}/output/LayerNorm/beta'
    for param in model.named_parameters():
        param_name = param[0]
        retriever = attrgetter(param_name)
        result = retriever(model)
        tf_name = param_mapping[param_name]
        value = torch.from_numpy(tf_data[tf_name])
        logger.info(f'TF: {tf_name}, PT: {param_name} ')
        if tf_name.endswith('/kernel'):
            if not tf_name.endswith('/intermediate/g_dense/kernel'):
                if not tf_name.endswith('/output/g_dense/kernel'):
                    value = value.T
        if tf_name.endswith('/depthwise_kernel'):
            value = value.permute(1, 2, 0)
        if tf_name.endswith('/pointwise_kernel'):
            value = value.permute(2, 1, 0)
        if tf_name.endswith('/conv_attn_key/bias'):
            value = value.unsqueeze(-1)
        result.data = value
    return model
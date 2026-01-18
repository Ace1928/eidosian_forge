from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...configuration_utils import PretrainedConfig
from ...generation import TFLogitsProcessorList
from ...modeling_tf_utils import (
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
def gather2d(target, id_tensor):
    idx = tf.stack([tf.range(tf.shape(id_tensor)[0], dtype=id_tensor.dtype), id_tensor[:, 0]], axis=-1)
    result = tf.gather_nd(target, idx)
    return tf.expand_dims(result, axis=-1)
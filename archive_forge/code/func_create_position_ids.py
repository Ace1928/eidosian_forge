from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
def create_position_ids(self, input_ids: tf.Tensor, inputs_embeds: tf.Tensor) -> tf.Tensor:
    if input_ids is None:
        return self.create_position_ids_from_inputs_embeds(inputs_embeds)
    else:
        return self.create_position_ids_from_input_ids(input_ids)
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_audio_spectrogram_transformer import ASTConfig
def get_shape(self, config):
    frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
    time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1
    return (frequency_out_dimension, time_out_dimension)
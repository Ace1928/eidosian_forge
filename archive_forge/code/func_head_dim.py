from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def head_dim(self):
    return self.hidden_size // self.num_attention_heads
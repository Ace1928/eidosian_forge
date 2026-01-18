from ...configuration_utils import PretrainedConfig
from ...utils import logging
@num_blocks.setter
def num_blocks(self, value):
    raise NotImplementedError('This model does not support the setting of `num_blocks`. Please set `block_sizes`.')
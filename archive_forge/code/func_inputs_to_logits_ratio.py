import functools
import operator
from ...configuration_utils import PretrainedConfig
from ...utils import logging
def inputs_to_logits_ratio(self):
    return functools.reduce(operator.mul, self.conv_stride, 1)
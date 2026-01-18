import functools
import operator
from ...configuration_utils import PretrainedConfig
from ...utils import logging
@property
def hidden_dropout(self):
    logger.warning_once('hidden_dropout is not used by the model and will be removed as config attribute in v4.35')
    return self._hidden_dropout
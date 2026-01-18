import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
@property
def inputs_for_other_tasks(self):
    return {'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask': {0: 'batch_size', 1: 'sequence_length'}}
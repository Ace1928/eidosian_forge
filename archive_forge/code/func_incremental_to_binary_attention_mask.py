from typing import Callable, List, Optional, Union
from urllib.parse import urlparse
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available
def incremental_to_binary_attention_mask(incremental_mask, num_classes=-1):
    if num_classes != -1:
        incremental_mask[incremental_mask >= num_classes] = -1
    negatives = incremental_mask == -1
    incremental_mask[negatives] = 0
    attn_mask = torch.nn.functional.one_hot(incremental_mask, num_classes=num_classes)
    attn_mask[negatives, :] = 0
    return attn_mask
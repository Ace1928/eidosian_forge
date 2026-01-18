import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from ..models.bert import BertTokenizer, BertTokenizerFast
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy
def numpy_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], np.ndarray) else first['label']
        dtype = np.int64 if isinstance(label, int) else np.float32
        batch['labels'] = np.array([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], np.ndarray):
            batch['labels'] = np.stack([f['label_ids'] for f in features])
        else:
            dtype = np.int64 if isinstance(first['label_ids'][0], int) else np.float32
            batch['labels'] = np.array([f['label_ids'] for f in features], dtype=dtype)
    for k, v in first.items():
        if k not in ('label', 'label_ids') and v is not None and (not isinstance(v, str)):
            if isinstance(v, np.ndarray):
                batch[k] = np.stack([f[k] for f in features])
            else:
                batch[k] = np.array([f[k] for f in features])
    return batch
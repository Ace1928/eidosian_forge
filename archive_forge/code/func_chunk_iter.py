from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Union
import numpy as np
import requests
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import is_torch_available, is_torchaudio_available, logging
from .audio_utils import ffmpeg_read
from .base import ChunkPipeline
def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right, dtype=None):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        processed = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors='pt')
        if dtype is not None:
            processed = processed.to(dtype=dtype)
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        is_last = chunk_end_idx > inputs_len if stride_right > 0 else chunk_end_idx >= inputs_len
        _stride_right = 0 if is_last else stride_right
        chunk_len = chunk.shape[0]
        stride = (chunk_len, _stride_left, _stride_right)
        if chunk.shape[0] > _stride_left:
            yield {'is_last': is_last, 'stride': stride, **processed}
        if is_last:
            break
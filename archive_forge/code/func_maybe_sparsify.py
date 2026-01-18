import logging
from pathlib import Path
from typing import Any, Callable, Dict, Set, Union
import torch
from xformers.utils import (
from ._sputnik_sparse import SparseCS
from .attention_mask import AttentionMask
from .base import Attention, AttentionConfig  # noqa
from .favor import FavorAttention  # noqa
from .global_tokens import GlobalAttention  # noqa
from .linformer import LinformerAttention  # noqa
from .local import LocalAttention  # noqa
from .nystrom import NystromAttention  # noqa
from .ortho import OrthoFormerAttention  # noqa
from .random import RandomAttention  # noqa
from .scaled_dot_product import ScaledDotProduct  # noqa
import_all_modules(str(Path(__file__).parent), "xformers.components.attention")
def maybe_sparsify(matrix) -> Any:
    if torch.count_nonzero(matrix).item() / matrix.numel() > _DENSITY_THRESHOLD:
        return AttentionMask.from_bool(matrix)
    return sparsify(matrix)
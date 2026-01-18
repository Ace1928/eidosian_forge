import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Tuple
import torch
import torchaudio
from torchaudio._internal import module_utils
from torchaudio.models import emformer_rnnt_base, RNNT, RNNTBeamSearch
def get_token_processor(self) -> TokenProcessor:
    """Constructs token processor.

        Returns:
            TokenProcessor
        """
    local_path = torchaudio.utils.download_asset(self._sp_model_path)
    return _SentencePieceTokenProcessor(local_path)
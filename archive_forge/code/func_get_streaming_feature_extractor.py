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
def get_streaming_feature_extractor(self) -> FeatureExtractor:
    """Constructs feature extractor for streaming (simultaneous) ASR.

        Returns:
            FeatureExtractor
        """
    local_path = torchaudio.utils.download_asset(self._global_stats_path)
    return _ModuleFeatureExtractor(torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, hop_length=self.hop_length), _FunctionalModule(lambda x: x.transpose(1, 0)), _FunctionalModule(lambda x: _piecewise_linear_log(x * _gain)), _GlobalStatsNormalization(local_path)))
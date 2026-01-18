from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
def denormalize(self, spectrogram):
    return self.normalize_min + (self.normalize_max - self.normalize_min) * ((spectrogram + 1) / 2)
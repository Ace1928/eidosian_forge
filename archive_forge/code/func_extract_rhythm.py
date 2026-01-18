import warnings
from typing import List, Optional, Union
import numpy
import numpy as np
from ...audio_utils import mel_filter_bank, spectrogram
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import (
def extract_rhythm(self, audio: np.ndarray):
    """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as
        tempo in bpm for an audio signal. For more information please visit
        https://essentia.upf.edu/reference/std_RhythmExtractor2013.html .

        Args:
            audio(`numpy.ndarray`):
                raw audio waveform which is passed to the Rhythm Extractor.
        """
    requires_backends(self, ['essentia'])
    essentia_tracker = essentia.standard.RhythmExtractor2013(method='multifeature')
    bpm, beat_times, confidence, estimates, essentia_beat_intervals = essentia_tracker(audio)
    return (bpm, beat_times, confidence, estimates, essentia_beat_intervals)
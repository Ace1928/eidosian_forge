import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy
def relative_batch_tokens_ids_to_notes(self, tokens: np.ndarray, beat_offset_idx: int, bars_per_batch: int, cutoff_time_idx: int):
    """
        Converts relative tokens to notes which are then used to generate pretty midi object.

        Args:
            tokens (`numpy.ndarray`):
                Tokens to be converted to notes.
            beat_offset_idx (`int`):
                Denotes beat offset index for each note in generated Midi.
            bars_per_batch (`int`):
                A parameter to control the Midi output generation.
            cutoff_time_idx (`int`):
                Denotes the cutoff time index for each note in generated Midi.
        """
    notes = None
    for index in range(len(tokens)):
        _tokens = tokens[index]
        _start_idx = beat_offset_idx + index * bars_per_batch * 4
        _cutoff_time_idx = cutoff_time_idx + _start_idx
        _notes = self.relative_tokens_ids_to_notes(_tokens, start_idx=_start_idx, cutoff_time_idx=_cutoff_time_idx)
        if len(_notes) == 0:
            pass
        elif notes is None:
            notes = _notes
        else:
            notes = np.concatenate((notes, _notes), axis=0)
    if notes is None:
        return []
    return notes
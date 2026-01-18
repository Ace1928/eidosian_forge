import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy
def relative_tokens_ids_to_notes(self, tokens: np.ndarray, start_idx: float, cutoff_time_idx: float=None):
    """
        Converts relative tokens to notes which will then be used to create Pretty Midi objects.

        Args:
            tokens (`numpy.ndarray`):
                Relative Tokens which will be converted to notes.
            start_idx (`float`):
                A parameter which denotes the starting index.
            cutoff_time_idx (`float`, *optional*):
                A parameter used while converting tokens to notes.
        """
    words = [self._convert_id_to_token(token) for token in tokens]
    current_idx = start_idx
    current_velocity = 0
    note_onsets_ready = [None for i in range(sum([k.endswith('NOTE') for k in self.encoder.keys()]) + 1)]
    notes = []
    for token_type, number in words:
        if token_type == 'TOKEN_SPECIAL':
            if number == 1:
                break
        elif token_type == 'TOKEN_TIME':
            current_idx = token_time_to_note(number=number, cutoff_time_idx=cutoff_time_idx, current_idx=current_idx)
        elif token_type == 'TOKEN_VELOCITY':
            current_velocity = number
        elif token_type == 'TOKEN_NOTE':
            notes = token_note_to_note(number=number, current_velocity=current_velocity, default_velocity=self.default_velocity, note_onsets_ready=note_onsets_ready, current_idx=current_idx, notes=notes)
        else:
            raise ValueError('Token type not understood!')
    for pitch, note_onset in enumerate(note_onsets_ready):
        if note_onset is not None:
            if cutoff_time_idx is None:
                cutoff = note_onset + 1
            else:
                cutoff = max(cutoff_time_idx, note_onset + 1)
            offset_idx = max(current_idx, cutoff)
            notes.append([note_onset, offset_idx, pitch, self.default_velocity])
    if len(notes) == 0:
        return []
    else:
        notes = np.array(notes)
        note_order = notes[:, 0] * 128 + notes[:, 1]
        notes = notes[note_order.argsort()]
        return notes
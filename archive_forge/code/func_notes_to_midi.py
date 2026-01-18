import json
import os
from typing import List, Optional, Tuple, Union
import numpy as np
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils import AddedToken, BatchEncoding, PaddingStrategy, PreTrainedTokenizer, TruncationStrategy
from ...utils import TensorType, is_pretty_midi_available, logging, requires_backends, to_numpy
def notes_to_midi(self, notes: np.ndarray, beatstep: np.ndarray, offset_sec: int=0.0):
    """
        Converts notes to Midi.

        Args:
            notes (`numpy.ndarray`):
                This is used to create Pretty Midi objects.
            beatstep (`numpy.ndarray`):
                This is the extrapolated beatstep that we get from feature extractor.
            offset_sec (`int`, *optional*, defaults to 0.0):
                This represents the offset seconds which is used while creating each Pretty Midi Note.
        """
    requires_backends(self, ['pretty_midi'])
    new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
    new_inst = pretty_midi.Instrument(program=0)
    new_notes = []
    for onset_idx, offset_idx, pitch, velocity in notes:
        new_note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=beatstep[onset_idx] - offset_sec, end=beatstep[offset_idx] - offset_sec)
        new_notes.append(new_note)
    new_inst.notes = new_notes
    new_pm.instruments.append(new_inst)
    new_pm.remove_invalid_notes()
    return new_pm
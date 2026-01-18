import datetime
from typing import Iterator, List, Optional, Union
import torch
from outlines.generate.generator import sequence_generator
def strip_stop_sequences(self, sequence: str, stop_sequences: Optional[List[str]]) -> str:
    """Remove the stop sequences from the generated sequences.

        Parameters
        ----------
        sequence
            One of the generated sequences.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        """
    if stop_sequences:
        match_indexes = [sequence.find(seq) for seq in stop_sequences]
        if any([index != -1 for index in match_indexes]):
            min_match_index_value = min([i for i in match_indexes if i != -1])
            min_match_index_pos = match_indexes.index(min_match_index_value)
            sequence = sequence[:match_indexes[min_match_index_pos] + len(stop_sequences[min_match_index_pos])]
    return sequence
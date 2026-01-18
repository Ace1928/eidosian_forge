import datetime
from typing import Iterator, List, Optional, Union
import torch
from outlines.generate.generator import sequence_generator
def is_stop_sequence_found(self, generated_sequences: List[str], stop_sequences: List[str]) -> bool:
    """Determine whether one of the stop sequences has been generated.

        Parameters
        ----------
        generated_sequences
            The list of sequences generated so far.
        stop_sequences
            The list that contains the sequence which stop the generation when
            found.

        Returns
        -------
        True if at least one of the stop sequences has been found in each generated
        sequence.

        """
    return all([any([seq in generated for seq in stop_sequences]) for generated in generated_sequences])
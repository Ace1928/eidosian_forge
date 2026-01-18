from __future__ import annotations
import math
from typing import List, NamedTuple, Union
import torch
import torchaudio
import torchaudio.lib.pybind11_prefixctc as cuctc
def cuda_ctc_decoder(tokens: Union[str, List[str]], nbest: int=1, beam_size: int=10, blank_skip_threshold: float=_DEFAULT_BLANK_SKIP_THREASHOLD) -> CUCTCDecoder:
    """Builds an instance of :class:`CUCTCDecoder`.

    Args:
        tokens (str or List[str]): File or list containing valid tokens.
            If using a file, the expected format is for tokens mapping to the same index to be on the same line
        beam_size (int, optional): The maximum number of hypos to hold after each decode step (Default: 10)
        nbest (int): The number of best decodings to return
        blank_id (int): The token ID corresopnding to the blank symbol.
        blank_skip_threshold (float): skip frames if log_prob(blank) > log(blank_skip_threshold), to speed up decoding
            (Default: 0.95).

    Returns:
        CUCTCDecoder: decoder

    Example
        >>> decoder = cuda_ctc_decoder(
        >>>     vocab_file="tokens.txt",
        >>>     blank_skip_threshold=0.95,
        >>> )
        >>> results = decoder(log_probs, encoder_out_lens) # List of shape (B, nbest) of Hypotheses
    """
    if type(tokens) == str:
        tokens = _get_vocab_list(tokens)
    return CUCTCDecoder(vocab_list=tokens, beam_size=beam_size, nbest=nbest, blank_skip_threshold=blank_skip_threshold)
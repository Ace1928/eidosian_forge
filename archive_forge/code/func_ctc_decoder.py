from __future__ import annotations
import itertools as it
from abc import abstractmethod
from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import torch
from flashlight.lib.text.decoder import (
from flashlight.lib.text.dictionary import (
from torchaudio.utils import download_asset
def ctc_decoder(lexicon: Optional[str], tokens: Union[str, List[str]], lm: Union[str, CTCDecoderLM]=None, lm_dict: Optional[str]=None, nbest: int=1, beam_size: int=50, beam_size_token: Optional[int]=None, beam_threshold: float=50, lm_weight: float=2, word_score: float=0, unk_score: float=float('-inf'), sil_score: float=0, log_add: bool=False, blank_token: str='-', sil_token: str='|', unk_word: str='<unk>') -> CTCDecoder:
    """Builds an instance of :class:`CTCDecoder`.

    Args:
        lexicon (str or None): lexicon file containing the possible words and corresponding spellings.
            Each line consists of a word and its space separated spelling. If `None`, uses lexicon-free
            decoding.
        tokens (str or List[str]): file or list containing valid tokens. If using a file, the expected
            format is for tokens mapping to the same index to be on the same line
        lm (str, CTCDecoderLM, or None, optional): either a path containing KenLM language model,
            custom language model of type `CTCDecoderLM`, or `None` if not using a language model
        lm_dict (str or None, optional): file consisting of the dictionary used for the LM, with a word
            per line sorted by LM index. If decoding with a lexicon, entries in lm_dict must also occur
            in the lexicon file. If `None`, dictionary for LM is constructed using the lexicon file.
            (Default: None)
        nbest (int, optional): number of best decodings to return (Default: 1)
        beam_size (int, optional): max number of hypos to hold after each decode step (Default: 50)
        beam_size_token (int, optional): max number of tokens to consider at each decode step.
            If `None`, it is set to the total number of tokens (Default: None)
        beam_threshold (float, optional): threshold for pruning hypothesis (Default: 50)
        lm_weight (float, optional): weight of language model (Default: 2)
        word_score (float, optional): word insertion score (Default: 0)
        unk_score (float, optional): unknown word insertion score (Default: -inf)
        sil_score (float, optional): silence insertion score (Default: 0)
        log_add (bool, optional): whether or not to use logadd when merging hypotheses (Default: False)
        blank_token (str, optional): token corresponding to blank (Default: "-")
        sil_token (str, optional): token corresponding to silence (Default: "|")
        unk_word (str, optional): word corresponding to unknown (Default: "<unk>")

    Returns:
        CTCDecoder: decoder

    Example
        >>> decoder = ctc_decoder(
        >>>     lexicon="lexicon.txt",
        >>>     tokens="tokens.txt",
        >>>     lm="kenlm.bin",
        >>> )
        >>> results = decoder(emissions) # List of shape (B, nbest) of Hypotheses
    """
    if lm_dict is not None and type(lm_dict) is not str:
        raise ValueError('lm_dict must be None or str type.')
    tokens_dict = _Dictionary(tokens)
    if lexicon:
        lexicon = _load_words(lexicon)
        decoder_options = _LexiconDecoderOptions(beam_size=beam_size, beam_size_token=beam_size_token or tokens_dict.index_size(), beam_threshold=beam_threshold, lm_weight=lm_weight, word_score=word_score, unk_score=unk_score, sil_score=sil_score, log_add=log_add, criterion_type=_CriterionType.CTC)
    else:
        decoder_options = _LexiconFreeDecoderOptions(beam_size=beam_size, beam_size_token=beam_size_token or tokens_dict.index_size(), beam_threshold=beam_threshold, lm_weight=lm_weight, sil_score=sil_score, log_add=log_add, criterion_type=_CriterionType.CTC)
    word_dict = _get_word_dict(lexicon, lm, lm_dict, tokens_dict, unk_word)
    if type(lm) == str:
        if _KenLM is None:
            raise RuntimeError('flashlight-text is installed, but KenLM is not installed. Please refer to https://github.com/kpu/kenlm#python-module for how to install it.')
        lm = _KenLM(lm, word_dict)
    elif lm is None:
        lm = _ZeroLM()
    return CTCDecoder(nbest=nbest, lexicon=lexicon, word_dict=word_dict, tokens_dict=tokens_dict, lm=lm, decoder_options=decoder_options, blank_token=blank_token, sil_token=sil_token, unk_word=unk_word)
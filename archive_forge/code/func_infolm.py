import os
from enum import unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from typing_extensions import Literal
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
def infolm(preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]], model_name_or_path: Union[str, os.PathLike]='bert-base-uncased', temperature: float=0.25, information_measure: _ALLOWED_INFORMATION_MEASURE_LITERAL='kl_divergence', idf: bool=True, alpha: Optional[float]=None, beta: Optional[float]=None, device: Optional[Union[str, torch.device]]=None, max_length: Optional[int]=None, batch_size: int=64, num_threads: int=0, verbose: bool=True, return_sentence_level_score: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Calculate `InfoLM`_ [1].

    InfoML corresponds to distance/divergence between predicted and reference sentence discrete distribution using
    one of the following information measures:

        - `KL divergence`_
        - `alpha divergence`_
        - `beta divergence`_
        - `AB divergence`_
        - `Rényi divergence`_
        - L1 distance
        - L2 distance
        - L-infinity distance
        - `Fisher-Rao distance`_

    `InfoLM`_ is a family of untrained embedding-based metrics which addresses some famous flaws of standard
    string-based metrics thanks to the usage of pre-trained masked language models. This family of metrics is mainly
    designed for summarization and data-to-text tasks.

    If you want to use IDF scaling over the whole dataset, please use the class metric.

    The implementation of this metric is fully based HuggingFace `transformers`' package.

    Args:
        preds:
            An iterable of hypothesis corpus.
        target:
            An iterable of reference corpus.
        model_name_or_path:
            A name or a model path used to load `transformers` pretrained model.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        information_measure:
            A name of information measure to be used. Please use one of: ['kl_divergence', 'alpha_divergence',
            'beta_divergence', 'ab_divergence', 'renyi_divergence', 'l1_distance', 'l2_distance', 'l_infinity_distance',
            'fisher_rao_distance']
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        alpha:
            Alpha parameter of the divergence used for alpha, AB and Rényi divergence measures.
        beta:
            Beta parameter of the divergence used for beta and AB divergence measures.
        device:
            A device to be used for calculation.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        batch_size:
            A batch size used for model processing.
        num_threads:
            A number of threads to use for a dataloader.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.
        return_sentence_level_score:
            An indication whether a sentence-level InfoLM score to be returned.

    Returns:
        A corpus-level InfoLM score.
        (Optionally) A list of sentence-level InfoLM scores if `return_sentence_level_score=True`.

    Example:
        >>> from torchmetrics.functional.text.infolm import infolm
        >>> preds = ['he read the book because he was interested in world history']
        >>> target = ['he was interested in world history because he read the book']
        >>> infolm(preds, target, model_name_or_path='google/bert_uncased_L-2_H-128_A-2', idf=False)
        tensor(-0.1784)

    References:
        [1] InfoLM: A New Metric to Evaluate Summarization & Data2Text Generation by Pierre Colombo, Chloé Clavel and
        Pablo Piantanida `InfoLM`_

    """
    tokenizer, model = _load_tokenizer_and_model(model_name_or_path, device)
    information_measure_cls = _InformationMeasure(information_measure, alpha, beta)
    max_length = max_length or model.config.max_length
    special_tokens_map = _get_special_tokens_map(tokenizer)
    preds_input_ids, preds_attention_mask, target_input_ids, target_attention_mask = _infolm_update(preds, target, tokenizer, max_length)
    preds_dataloader = _get_dataloader(preds_input_ids, preds_attention_mask, idf, batch_size, num_threads)
    target_dataloader = _get_dataloader(target_input_ids, target_attention_mask, idf, batch_size, num_threads)
    info_lm_score = _infolm_compute(model, preds_dataloader, target_dataloader, temperature, idf, information_measure_cls, special_tokens_map, verbose)
    if return_sentence_level_score:
        return (info_lm_score.mean(), info_lm_score)
    return info_lm_score.mean()
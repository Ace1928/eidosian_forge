import os
from typing import Dict, List, Union
import tensorflow as tf
from keras_nlp.tokenizers import BytePairTokenizer
from tensorflow_text import pad_model_inputs
from ...modeling_tf_utils import keras
from .tokenization_gpt2 import GPT2Tokenizer
@classmethod
def from_tokenizer(cls, tokenizer: GPT2Tokenizer, *args, **kwargs):
    """Creates TFGPT2Tokenizer from GPT2Tokenizer

        Args:
            tokenizer (GPT2Tokenizer)

        Examples:

        ```python
        from transformers import AutoTokenizer, TFGPT2Tokenizer

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tf_tokenizer = TFGPT2Tokenizer.from_tokenizer(tokenizer)
        ```
        """
    merges = [' '.join(m) for m in tokenizer.bpe_ranks.keys()]
    vocab = tokenizer.get_vocab()
    return cls(vocab, merges, *args, **kwargs)
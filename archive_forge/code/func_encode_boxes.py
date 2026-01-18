import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union
from ...tokenization_utils_base import (
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, is_sentencepiece_available, logging
def encode_boxes(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]=None, boxes: Optional[List[List[int]]]=None, word_labels: Optional[List[List[int]]]=None, add_special_tokens: bool=True, padding: Union[bool, str, PaddingStrategy]=False, truncation: Union[bool, str, TruncationStrategy]=None, max_length: Optional[int]=None, stride: int=0, return_tensors: Optional[Union[str, TensorType]]=None, **kwargs) -> List[int]:
    """
        Args:
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary. Same as doing
        `self.convert_tokens_to_ids(self.tokenize(text))`.
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
    encoded_inputs = self.encode_plus_boxes(text, text_pair=text_pair, boxes=boxes, word_labels=word_labels, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation, max_length=max_length, stride=stride, return_tensors=return_tensors, **kwargs)
    return encoded_inputs['input_ids']
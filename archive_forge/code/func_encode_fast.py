import os
import re
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_torch_available, logging
def encode_fast(self, text: Union[str, List[str]], return_tensors: Union[str, bool]=False) -> Union[List[int], List[List[int]], 'torch.Tensor']:
    """
        Encodes a text or batch of texts to token ids using preprocessing and the raw SP tokenizer. This has reduced
        functionality but is often much faster.

        Does NOT handle special tokens correctly, these can manually be added as ids afterwards.

        Does NOT support padding, these can manually be added as ids afterwards.

        Use default HuggingFace tokenization methods for full functionality.

        Args:
            text (`str` or `List[str]`): One or several text(s) to convert to token ids.
            return_tensors (`str` or `bool`): Returns PyTorch tensors if set to True or "pt"

        Returns:
            `List[int]`, `List[List[int]]`, or `torch.Tensor`: The encoded text(s) as token ids.
        """
    if isinstance(text, str):
        text = self.preprocess_text(text)
        token_ids = self.sp_model.encode(text)
    else:
        text = [self.preprocess_text(t) for t in text]
        token_ids = self.sp_model.encode(text)
    if return_tensors is True or return_tensors == 'pt':
        token_ids = torch.tensor(token_ids)
    return token_ids
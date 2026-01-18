import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
def append_segment(self, text, tokens, system=True):
    """
        Append a new segment to the history.

        args:
            text (`str`): The text of the new segment.
            tokens (`torch.LongTensor`): The tokens of the new segment.
            system (`bool`, *optional*): Whether the new segment is a system or user segment.
        """
    if len(text) == 0 or len(tokens) == 0:
        raise ValueError("Can't append empty text or token list to history.")
    original_text_length = len(self.text)
    self.text += text
    self.text_spans.append((original_text_length, len(self.text)))
    self.system_spans.append(system)
    original_token_length = len(self.tokens)
    self.tokens = torch.cat((self.tokens, tokens))
    if system:
        self.token_masks = torch.cat((self.token_masks, torch.zeros_like(tokens)))
    else:
        self.token_masks = torch.cat((self.token_masks, torch.ones_like(tokens)))
    self.token_spans.append((original_token_length, len(self.tokens)))
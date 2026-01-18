import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
def show_tokens(self, tokenizer, show_legend=False):
    """
        Print the history tokens.
        """
    if not is_rich_available():
        warnings.warn('install rich to display tokens')
        return
    text = Text()
    prompt_end = self.token_spans[0][1]
    for i, (token, mask) in enumerate(zip(self.tokens, self.token_masks)):
        if i < prompt_end:
            text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.prompt_color)
            text.append(' ')
        elif mask == 0:
            text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.system_color)
            text.append(' ')
        else:
            text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.model_color)
            text.append(' ')
    text.append(f'\n\nReward: {self.reward}', style=self.reward_color)
    print(text)
    if show_legend:
        self.show_colour_legend()
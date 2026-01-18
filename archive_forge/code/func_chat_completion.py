import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
def chat_completion(self, dialogs: List[Dialog], temperature: float=0.6, top_p: float=0.9, max_gen_len: Optional[int]=None, logprobs: bool=False) -> List[ChatPrediction]:
    """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
    if max_gen_len is None:
        max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = [self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs]
    generation_tokens, generation_logprobs = self.generate(prompt_tokens=prompt_tokens, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs)
    if logprobs:
        return [{'generation': {'role': 'assistant', 'content': self.tokenizer.decode(t)}, 'tokens': [self.tokenizer.decode([x]) for x in t], 'logprobs': logprobs_i} for t, logprobs_i in zip(generation_tokens, generation_logprobs)]
    return [{'generation': {'role': 'assistant', 'content': self.tokenizer.decode(t)}} for t in generation_tokens]
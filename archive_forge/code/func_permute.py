import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Config, LlamaConfig
from einops import rearrange
def permute(w):
    return rearrange(w, '(h d two) n -> (h two d) n', d=config.n_embd // config.n_head // 2, two=2)
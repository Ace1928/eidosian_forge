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
def key_mapping_emb(key):
    return re.sub('^transformer.embeddings.word_embeddings.', 'model.embed_tokens.', key)
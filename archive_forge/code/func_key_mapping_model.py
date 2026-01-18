import math
import re
from collections import OrderedDict
import torch
import torch.nn.functional as F
from transformers import GPT2Config, OPTConfig
def key_mapping_model(key):
    key = re.sub('^model.decoder.', 'transformer.', key)
    key = re.sub('^decoder.', 'transformer.', key)
    return key
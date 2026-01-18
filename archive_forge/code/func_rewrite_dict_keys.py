import argparse
import json
import os
import re
from collections import OrderedDict
from os.path import basename, dirname
import fairseq
import torch
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary
from transformers import FSMTConfig, FSMTForConditionalGeneration
from transformers.models.fsmt.tokenization_fsmt import VOCAB_FILES_NAMES
from transformers.tokenization_utils_base import TOKENIZER_CONFIG_FILE
from transformers.utils import WEIGHTS_NAME, logging
def rewrite_dict_keys(d):
    d2 = dict(((re.sub('@@$', '', k), v) if k.endswith('@@') else (re.sub('$', '</w>', k), v) for k, v in d.items()))
    keep_keys = '<s> <pad> </s> <unk>'.split()
    for k in keep_keys:
        del d2[f'{k}</w>']
        d2[k] = d[k]
    return d2
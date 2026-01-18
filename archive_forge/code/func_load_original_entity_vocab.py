import argparse
import json
import os
from collections import OrderedDict
import torch
from transformers import LukeConfig, LukeForMaskedLM, MLukeTokenizer, XLMRobertaTokenizer
from transformers.tokenization_utils_base import AddedToken
def load_original_entity_vocab(entity_vocab_path):
    SPECIAL_TOKENS = ['[MASK]', '[PAD]', '[UNK]']
    data = [json.loads(line) for line in open(entity_vocab_path)]
    new_mapping = {}
    for entry in data:
        entity_id = entry['id']
        for entity_name, language in entry['entities']:
            if entity_name in SPECIAL_TOKENS:
                new_mapping[entity_name] = entity_id
                break
            new_entity_name = f'{language}:{entity_name}'
            new_mapping[new_entity_name] = entity_id
    return new_mapping
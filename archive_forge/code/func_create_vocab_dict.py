import argparse
import json
import os
import fairseq
import torch
from torch import nn
from transformers import (
def create_vocab_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        words = [line.split(' ')[0] for line in lines]
    num_words = len(words)
    vocab_dict = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3}
    vocab_dict.update(dict(zip(words, range(4, num_words + 4))))
    return vocab_dict
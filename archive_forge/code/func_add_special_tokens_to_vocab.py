import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
def add_special_tokens_to_vocab(model_dir: Path, separate_vocab=False) -> None:
    if separate_vocab:
        vocab = load_yaml(find_src_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        num_added = add_to_vocab_(vocab, ['<pad>'])
        save_json(vocab, model_dir / 'vocab.json')
        vocab = load_yaml(find_tgt_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        num_added = add_to_vocab_(vocab, ['<pad>'])
        save_json(vocab, model_dir / 'target_vocab.json')
        save_tokenizer_config(model_dir, separate_vocabs=separate_vocab)
    else:
        vocab = load_yaml(find_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        num_added = add_to_vocab_(vocab, ['<pad>'])
        print(f'added {num_added} tokens to vocab')
        save_json(vocab, model_dir / 'vocab.json')
        save_tokenizer_config(model_dir)
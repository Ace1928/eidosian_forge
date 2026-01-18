import argparse
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory
import esm as esm_module
import torch
from esm.esmfold.v1.misc import batch_encode_sequences as esmfold_encode_sequences
from esm.esmfold.v1.pretrained import esmfold_v1
from transformers.models.esm.configuration_esm import EsmConfig, EsmFoldConfig
from transformers.models.esm.modeling_esm import (
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
from transformers.models.esm.tokenization_esm import EsmTokenizer
from transformers.utils import logging
def get_esmfold_tokenizer():
    with TemporaryDirectory() as tempdir:
        vocab = '\n'.join(restypes_with_extras)
        vocab_file = Path(tempdir) / 'vocab.txt'
        vocab_file.write_text(vocab)
        hf_tokenizer = EsmTokenizer(vocab_file=str(vocab_file))
    hf_tokenizer.pad_token_id = 0
    return hf_tokenizer
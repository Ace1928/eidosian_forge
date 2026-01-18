import json
import os
from typing import Iterator, List, Optional, Union, Tuple
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.models import Unigram
from .base_tokenizer import BaseTokenizer
@staticmethod
def from_spm(filename: str):
    try:
        import sys
        sys.path.append('.')
        import sentencepiece_model_pb2 as model
    except Exception:
        raise Exception("You don't seem to have the required protobuf file, in order to use this function you need to run `pip install protobuf` and `wget https://raw.githubusercontent.com/google/sentencepiece/master/python/src/sentencepiece/sentencepiece_model_pb2.py` for us to be able to read the intrinsics of your spm_file. `pip install sentencepiece` is not required.")
    m = model.ModelProto()
    m.ParseFromString(open(filename, 'rb').read())
    precompiled_charsmap = m.normalizer_spec.precompiled_charsmap
    vocab = [(piece.piece, piece.score) for piece in m.pieces]
    unk_id = m.trainer_spec.unk_id
    model_type = m.trainer_spec.model_type
    byte_fallback = m.trainer_spec.byte_fallback
    if model_type != 1:
        raise Exception("You're trying to run a `Unigram` model but you're file was trained with a different algorithm")
    replacement = '▁'
    add_prefix_space = True
    tokenizer = Tokenizer(Unigram(vocab, unk_id, byte_fallback))
    if precompiled_charsmap:
        tokenizer.normalizer = normalizers.Sequence([normalizers.Precompiled(precompiled_charsmap), normalizers.Replace(Regex(' {2,}'), ' ')])
    else:
        tokenizer.normalizer = normalizers.Sequence([normalizers.Replace(Regex(' {2,}'), ' ')])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)
    tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)
    parameters = {'model': 'SentencePieceUnigram'}
    obj = BaseTokenizer.__new__(SentencePieceUnigramTokenizer, tokenizer, parameters)
    BaseTokenizer.__init__(obj, tokenizer, parameters)
    return obj
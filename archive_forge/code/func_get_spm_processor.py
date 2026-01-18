import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def get_spm_processor(self, from_slow=False):
    tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
    if self.legacy or from_slow:
        tokenizer.Load(self.vocab_file)
        return tokenizer
    with open(self.vocab_file, 'rb') as f:
        sp_model = f.read()
        model_pb2 = import_protobuf(f'The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)')
        model = model_pb2.ModelProto.FromString(sp_model)
        normalizer_spec = model_pb2.NormalizerSpec()
        normalizer_spec.add_dummy_prefix = False
        model.normalizer_spec.MergeFrom(normalizer_spec)
        sp_model = model.SerializeToString()
        tokenizer.LoadFromSerializedProto(sp_model)
    return tokenizer
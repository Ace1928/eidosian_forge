import warnings
from typing import Dict, List, Tuple
from packaging import version
from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram, WordPiece
from .utils import is_protobuf_available, requires_backends
from .utils.import_utils import PROTOBUF_IMPORT_ERROR
def pre_tokenizer(self, replacement, add_prefix_space):
    return None
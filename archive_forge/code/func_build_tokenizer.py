from abc import ABC
from abc import abstractmethod
from tokenizers import Tokenizer
from rwkv_tokenizer import RWKV_TOKENIZER, TRIE_TOKENIZER
from typing import List, Union
def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type), flush=True)
    if args.tokenizer_type.lower() == 'HFTokenizer'.lower():
        assert args.vocab_file is not None
        tokenizer = HFTokenizer(args.vocab_file)
    elif args.tokenizer_type.lower() == 'RWKVTokenizer'.lower():
        assert args.vocab_file is not None
        tokenizer = RWKVTokenizer(args.vocab_file)
    else:
        raise NotImplementedError('{} tokenizer is not implemented.'.format(args.tokenizer_type))
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)
    return tokenizer
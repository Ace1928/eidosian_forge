from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, EncodeInput, Encoding, InputSequence, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
def enable_truncation(self, max_length: int, stride: Optional[int]=0, strategy: Optional[str]='longest_first'):
    """Change the truncation options

        Args:
            max_length: unsigned int:
                The maximum length at which to truncate

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence

            strategy: (`optional`) str:
                Can be one of `longest_first`, `only_first` or `only_second`
        """
    return self._tokenizer.enable_truncation(max_length, stride=stride, strategy=strategy)
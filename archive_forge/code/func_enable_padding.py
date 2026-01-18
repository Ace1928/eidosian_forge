from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, EncodeInput, Encoding, InputSequence, Tokenizer
from tokenizers.decoders import Decoder
from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
def enable_padding(self, direction: Optional[str]='right', pad_to_multiple_of: Optional[int]=None, pad_id: Optional[int]=0, pad_type_id: Optional[int]=0, pad_token: Optional[str]='[PAD]', length: Optional[int]=None):
    """Change the padding strategy

        Args:
            direction: (`optional`) str:
                Can be one of: `right` or `left`

            pad_to_multiple_of: (`optional`) unsigned int:
                If specified, the padding length should always snap to the next multiple of
                the given value. For example if we were going to pad with a length of 250 but
                `pad_to_multiple_of=8` then we will pad to 256.

            pad_id: (`optional`) unsigned int:
                The indice to be used when padding

            pad_type_id: (`optional`) unsigned int:
                The type indice to be used when padding

            pad_token: (`optional`) str:
                The pad token to be used when padding

            length: (`optional`) unsigned int:
                If specified, the length at which to pad. If not specified
                we pad using the size of the longest sequence in a batch
        """
    return self._tokenizer.enable_padding(direction=direction, pad_to_multiple_of=pad_to_multiple_of, pad_id=pad_id, pad_type_id=pad_type_id, pad_token=pad_token, length=length)
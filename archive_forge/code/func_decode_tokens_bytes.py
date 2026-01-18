import collections
from typing import Optional
import regex
import tiktoken
def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
    """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
    return [self._decoder[token] for token in tokens]
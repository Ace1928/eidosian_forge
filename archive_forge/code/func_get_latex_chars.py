import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
def get_latex_chars(self, unicode_: str, final: bool=False) -> Iterator[str]:
    if not isinstance(unicode_, str):
        raise TypeError('expected unicode for encode input, but got {0} instead'.format(unicode_.__class__.__name__))
    for pos, c in enumerate(unicode_):
        bytes_, tokens = self._get_latex_chars_tokens_from_char(c)
        space, bytes_ = self.get_space_bytes(bytes_)
        if tokens and tokens[-1].name == 'control_word':
            self.state = 'S'
        elif tokens:
            self.state = 'M'
        if space:
            yield space
        yield bytes_
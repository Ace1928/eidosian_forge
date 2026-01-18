import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
def find_latex(encoding: str) -> Optional[CodecInfo]:
    """Return a :class:`codecs.CodecInfo` instance for the requested
    LaTeX *encoding*, which must be equal to ``latex``,
    or to ``latex+<encoding>``
    where ``<encoding>`` describes another encoding.
    """
    IncEnc: Type[LatexIncrementalEncoder]
    IncDec: Type[LatexIncrementalDecoder]
    if '_' in encoding:
        encoding, _, inputenc_ = encoding.partition('_')
    else:
        encoding, _, inputenc_ = encoding.partition('+')
    if not inputenc_:
        inputenc_ = 'ascii'
    if encoding == 'latex':
        incremental_encoder = type('incremental_encoder', (LatexIncrementalEncoder,), dict(inputenc=inputenc_))
        incremental_decoder = type('incremental_encoder', (LatexIncrementalDecoder,), dict(inputenc=inputenc_))
    elif encoding == 'ulatex':
        incremental_encoder = type('incremental_encoder', (UnicodeLatexIncrementalEncoder,), dict(inputenc=inputenc_))
        incremental_decoder = type('incremental_encoder', (UnicodeLatexIncrementalDecoder,), dict(inputenc=inputenc_))
    else:
        return None

    class Codec(LatexCodec):
        IncrementalEncoder = incremental_encoder
        IncrementalDecoder = incremental_decoder

    class StreamWriter(Codec, codecs.StreamWriter):
        pass

    class StreamReader(Codec, codecs.StreamReader):
        pass
    return codecs.CodecInfo(encode=Codec().encode, decode=Codec().decode, incrementalencoder=Codec.IncrementalEncoder, incrementaldecoder=Codec.IncrementalDecoder, streamreader=StreamReader, streamwriter=StreamWriter)
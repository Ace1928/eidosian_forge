from .core import encode, decode, alabel, ulabel, IDNAError
import codecs
import re
from typing import Any, Tuple, Optional
def search_function(name: str) -> Optional[codecs.CodecInfo]:
    if name != 'idna2008':
        return None
    return codecs.CodecInfo(name=name, encode=Codec().encode, decode=Codec().decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamwriter=StreamWriter, streamreader=StreamReader)
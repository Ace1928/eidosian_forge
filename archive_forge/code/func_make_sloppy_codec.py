from __future__ import annotations
import codecs
from encodings import normalize_encoding
import sys
def make_sloppy_codec(encoding):
    """
    Take a codec name, and return a 'sloppy' version of that codec that can
    encode and decode the unassigned bytes in that encoding.

    Single-byte encodings in the standard library are defined using some
    boilerplate classes surrounding the functions that do the actual work,
    `codecs.charmap_decode` and `charmap_encode`. This function, given an
    encoding name, *defines* those boilerplate classes.
    """
    all_bytes = bytes(range(256))
    sloppy_chars = list(all_bytes.decode('latin-1'))
    if PY26:
        decoded_chars = all_bytes.decode(encoding, 'replace')
    else:
        decoded_chars = all_bytes.decode(encoding, errors='replace')
    for i, char in enumerate(decoded_chars):
        if char != REPLACEMENT_CHAR:
            sloppy_chars[i] = char
    sloppy_chars[26] = REPLACEMENT_CHAR
    decoding_table = ''.join(sloppy_chars)
    encoding_table = codecs.charmap_build(decoding_table)

    class Codec(codecs.Codec):

        def encode(self, input, errors='strict'):
            return codecs.charmap_encode(input, errors, encoding_table)

        def decode(self, input, errors='strict'):
            return codecs.charmap_decode(input, errors, decoding_table)

    class IncrementalEncoder(codecs.IncrementalEncoder):

        def encode(self, input, final=False):
            return codecs.charmap_encode(input, self.errors, encoding_table)[0]

    class IncrementalDecoder(codecs.IncrementalDecoder):

        def decode(self, input, final=False):
            return codecs.charmap_decode(input, self.errors, decoding_table)[0]

    class StreamWriter(Codec, codecs.StreamWriter):
        pass

    class StreamReader(Codec, codecs.StreamReader):
        pass
    return codecs.CodecInfo(name='sloppy-' + encoding, encode=Codec().encode, decode=Codec().decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamreader=StreamReader, streamwriter=StreamWriter)
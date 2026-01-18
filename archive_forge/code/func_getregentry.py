from .core import encode, decode, alabel, ulabel, IDNAError
import codecs
import re
def getregentry():
    return codecs.CodecInfo(name='idna', encode=Codec().encode, decode=Codec().decode, incrementalencoder=IncrementalEncoder, incrementaldecoder=IncrementalDecoder, streamwriter=StreamWriter, streamreader=StreamReader)
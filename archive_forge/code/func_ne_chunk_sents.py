from nltk.chunk.api import ChunkParserI
from nltk.chunk.regexp import RegexpChunkParser, RegexpParser
from nltk.chunk.util import (
from nltk.data import load
def ne_chunk_sents(tagged_sentences, binary=False):
    """
    Use NLTK's currently recommended named entity chunker to chunk the
    given list of tagged sentences, each consisting of a list of tagged tokens.
    """
    if binary:
        chunker_pickle = _BINARY_NE_CHUNKER
    else:
        chunker_pickle = _MULTICLASS_NE_CHUNKER
    chunker = load(chunker_pickle)
    return chunker.parse_sents(tagged_sentences)
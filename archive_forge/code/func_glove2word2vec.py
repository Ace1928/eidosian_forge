import sys
import logging
import argparse
from gensim import utils
from gensim.utils import deprecated
from gensim.models.keyedvectors import KeyedVectors
@deprecated('KeyedVectors.load_word2vec_format(.., binary=False, no_header=True) loads GLoVE text vectors.')
def glove2word2vec(glove_input_file, word2vec_output_file):
    """Convert `glove_input_file` in GloVe format to word2vec format and write it to `word2vec_output_file`.

    Parameters
    ----------
    glove_input_file : str
        Path to file in GloVe format.
    word2vec_output_file: str
        Path to output file.

    Returns
    -------
    (int, int)
        Number of vectors (lines) of input file and its dimension.

    """
    glovekv = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)
    num_lines, num_dims = (len(glovekv), glovekv.vector_size)
    logger.info('converting %i vectors from %s to %s', num_lines, glove_input_file, word2vec_output_file)
    glovekv.save_word2vec_format(word2vec_output_file, binary=False)
    return (num_lines, num_dims)
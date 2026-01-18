import sys
import logging
import argparse
from gensim import utils
from gensim.utils import deprecated
from gensim.models.keyedvectors import KeyedVectors
def get_glove_info(glove_file_name):
    """Get number of vectors in provided `glove_file_name` and dimension of vectors.

    Parameters
    ----------
    glove_file_name : str
        Path to file in GloVe format.

    Returns
    -------
    (int, int)
        Number of vectors (lines) of input file and its dimension.

    """
    with utils.open(glove_file_name, 'rb') as f:
        num_lines = sum((1 for _ in f))
    with utils.open(glove_file_name, 'rb') as f:
        num_dims = len(f.readline().split()) - 1
    return (num_lines, num_dims)
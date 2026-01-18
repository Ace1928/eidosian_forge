from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
def process_parsed_fock_matrix(fock_matrix):
    """The Fock matrix is parsed as a list, while it should actually be
    a square matrix, this function takes the list of finds the right dimensions
    in order to reshape the matrix.
    """
    total_elements = len(fock_matrix)
    n_rows = int(np.sqrt(total_elements))
    n_cols = n_rows
    chunks = 6 * n_rows
    chunk_indices = np.arange(chunks, total_elements, chunks)
    fock_matrix_chunks = np.split(fock_matrix, chunk_indices)
    fock_matrix_reshaped = np.zeros(shape=(n_rows, n_cols), dtype=float)
    index_cols = 0
    for fock_matrix_chunk in fock_matrix_chunks:
        n_cols_chunks = len(fock_matrix_chunk) / n_rows
        n_cols_chunks = int(n_cols_chunks)
        fock_matrix_chunk_reshaped = np.reshape(fock_matrix_chunk, (n_rows, n_cols_chunks))
        fock_matrix_reshaped[:, index_cols:index_cols + n_cols_chunks] = fock_matrix_chunk_reshaped
        index_cols += n_cols_chunks
    return fock_matrix_reshaped
import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32
def save_sparse_csr(filename, matrix, metadata=None):
    data = {'data': matrix.data, 'indices': matrix.indices, 'indptr': matrix.indptr, 'shape': matrix.shape, 'metadata': metadata}
    np.savez(filename, **data)
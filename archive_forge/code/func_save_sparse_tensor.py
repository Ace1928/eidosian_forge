import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32
def save_sparse_tensor(filename, matrix, metadata=None):
    data = {'indices': matrix._indices(), 'values': matrix._values(), 'size': matrix.size(), 'metadata': metadata}
    torch.save(data, filename)
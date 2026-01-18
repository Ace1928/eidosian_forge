def spmatrix_repr_default(X):
    return "<%ix%i sparse matrix, tc='%c', nnz=%i>" % (X.size[0], X.size[1], X.typecode, len(X.V))
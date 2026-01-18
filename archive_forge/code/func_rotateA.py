import numpy as np
def rotateA(A, T, rotation_method='orthogonal'):
    """
    For orthogonal rotation methods :math:`L=AT`, where :math:`T` is an
    orthogonal matrix. For oblique rotation matrices :math:`L=A(T^*)^{-1}`,
    where :math:`T` is a normal matrix, i.e., :math:`TT^*=T^*T`. Oblique
    rotations relax the orthogonality constraint in order to gain simplicity
    in the interpretation.
    """
    if rotation_method == 'orthogonal':
        L = A.dot(T)
    elif rotation_method == 'oblique':
        L = A.dot(np.linalg.inv(T.T))
    else:
        raise ValueError('rotation_method should be one of {orthogonal, oblique}')
    return L
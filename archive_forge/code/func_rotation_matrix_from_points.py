import numpy as np
def rotation_matrix_from_points(m0, m1):
    """Returns a rigid transformation/rotation matrix that minimizes the
    RMSD between two set of points.
    
    m0 and m1 should be (3, npoints) numpy arrays with
    coordinates as columns::

        (x1  x2   x3   ... xN
         y1  y2   y3   ... yN
         z1  z2   z3   ... zN)

    The centeroids should be set to origin prior to
    computing the rotation matrix.

    The rotation matrix is computed using quaternion
    algebra as detailed in::
        
        Melander et al. J. Chem. Theory Comput., 2015, 11,1055
    """
    v0 = np.copy(m0)
    v1 = np.copy(m1)
    R11, R22, R33 = np.sum(v0 * v1, axis=1)
    R12, R23, R31 = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
    R13, R21, R32 = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
    f = [[R11 + R22 + R33, R23 - R32, R31 - R13, R12 - R21], [R23 - R32, R11 - R22 - R33, R12 + R21, R13 + R31], [R31 - R13, R12 + R21, -R11 + R22 - R33, R23 + R32], [R12 - R21, R13 + R31, R23 + R32, -R11 - R22 + R33]]
    F = np.array(f)
    w, V = np.linalg.eigh(F)
    q = V[:, np.argmax(w)]
    R = quaternion_to_matrix(q)
    return R
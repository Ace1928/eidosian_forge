import numpy as np
def quaternion_to_matrix(q):
    """Returns a rotation matrix.
    
    Computed from a unit quaternion Input as (4,) numpy array.
    """
    q0, q1, q2, q3 = q
    R_q = [[q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)], [2 * (q1 * q2 + q0 * q3), q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * (q2 * q3 - q0 * q1)], [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2]]
    return np.array(R_q)
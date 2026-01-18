import numpy as np
from Bio.PDB.PDBExceptions import PDBException
def qcp(coords1, coords2, natoms):
    """Implement the QCP code in Python.

    Input coordinate arrays must be centered at the origin and have
    shape Nx3.

    Variable names match (as much as possible) the C implementation.
    """
    G1 = np.trace(np.dot(coords2, coords2.T))
    G2 = np.trace(np.dot(coords1, coords1.T))
    A = np.dot(coords2.T, coords1)
    E0 = (G1 + G2) * 0.5
    Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz = A.flatten()
    Sxx2 = Sxx * Sxx
    Syy2 = Syy * Syy
    Szz2 = Szz * Szz
    Sxy2 = Sxy * Sxy
    Syz2 = Syz * Syz
    Sxz2 = Sxz * Sxz
    Syx2 = Syx * Syx
    Szy2 = Szy * Szy
    Szx2 = Szx * Szx
    SyzSzymSyySzz2 = 2.0 * (Syz * Szy - Syy * Szz)
    Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2
    C2 = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2)
    C1 = 8.0 * (Sxx * Syz * Szy + Syy * Szx * Sxz + Szz * Sxy * Syx - Sxx * Syy * Szz - Syz * Szx * Sxy - Szy * Syx * Sxz)
    SxzpSzx = Sxz + Szx
    SyzpSzy = Syz + Szy
    SxypSyx = Sxy + Syx
    SyzmSzy = Syz - Szy
    SxzmSzx = Sxz - Szx
    SxymSyx = Sxy - Syx
    SxxpSyy = Sxx + Syy
    SxxmSyy = Sxx - Syy
    Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2
    negSxzpSzx = -SxzpSzx
    negSxzmSzx = -SxzmSzx
    negSxymSyx = -SxymSyx
    SxxpSyy_p_Szz = SxxpSyy + Szz
    C0 = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2 + (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2) + (negSxzpSzx * SyzmSzy + SxymSyx * (SxxmSyy - Szz)) * (negSxzmSzx * SyzpSzy + SxymSyx * (SxxmSyy + Szz)) + (negSxzpSzx * SyzpSzy - SxypSyx * (SxxpSyy - Szz)) * (negSxzmSzx * SyzmSzy - SxypSyx * SxxpSyy_p_Szz) + (+SxypSyx * SyzpSzy + SxzpSzx * (SxxmSyy + Szz)) * (negSxymSyx * SyzmSzy + SxzpSzx * SxxpSyy_p_Szz) + (+SxypSyx * SyzmSzy + SxzmSzx * (SxxmSyy - Szz)) * (negSxymSyx * SyzpSzy + SxzmSzx * (SxxpSyy - Szz))
    nr_it = 50
    mxEigenV = E0
    evalprec = 1e-11
    for _ in range(nr_it):
        oldg = mxEigenV
        x2 = mxEigenV * mxEigenV
        b = (x2 + C2) * mxEigenV
        a = b + C1
        f = a * mxEigenV + C0
        f_prime = 2.0 * x2 * mxEigenV + b + a
        delta = f / (f_prime + evalprec)
        mxEigenV = abs(mxEigenV - delta)
        if mxEigenV - oldg < evalprec * mxEigenV:
            break
    else:
        print(f'Newton-Rhapson did not converge after {nr_it} iterations')
    rmsd = (2.0 * abs(E0 - mxEigenV) / natoms) ** 0.5
    a11 = SxxpSyy + Szz - mxEigenV
    a12 = SyzmSzy
    a13 = negSxzmSzx
    a14 = SxymSyx
    a21 = SyzmSzy
    a22 = SxxmSyy - Szz - mxEigenV
    a23 = SxypSyx
    a24 = SxzpSzx
    a31 = a13
    a32 = a23
    a33 = Syy - Sxx - Szz - mxEigenV
    a34 = SyzpSzy
    a41 = a14
    a42 = a24
    a43 = a34
    a44 = Szz - SxxpSyy - mxEigenV
    a3344_4334 = a33 * a44 - a43 * a34
    a3244_4234 = a32 * a44 - a42 * a34
    a3243_4233 = a32 * a43 - a42 * a33
    a3143_4133 = a31 * a43 - a41 * a33
    a3144_4134 = a31 * a44 - a41 * a34
    a3142_4132 = a31 * a42 - a41 * a32
    q1 = a22 * a3344_4334 - a23 * a3244_4234 + a24 * a3243_4233
    q2 = -a21 * a3344_4334 + a23 * a3144_4134 - a24 * a3143_4133
    q3 = a21 * a3244_4234 - a22 * a3144_4134 + a24 * a3142_4132
    q4 = -a21 * a3243_4233 + a22 * a3143_4133 - a23 * a3142_4132
    qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4
    evecprec = 1e-06
    if qsqr < evecprec:
        q1 = a12 * a3344_4334 - a13 * a3244_4234 + a14 * a3243_4233
        q2 = -a11 * a3344_4334 + a13 * a3144_4134 - a14 * a3143_4133
        q3 = a11 * a3244_4234 - a12 * a3144_4134 + a14 * a3142_4132
        q4 = -a11 * a3243_4233 + a12 * a3143_4133 - a13 * a3142_4132
        qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4
        if qsqr < evecprec:
            a1324_1423 = a13 * a24 - a14 * a23
            a1224_1422 = a12 * a24 - a14 * a22
            a1223_1322 = a12 * a23 - a13 * a22
            a1124_1421 = a11 * a24 - a14 * a21
            a1123_1321 = a11 * a23 - a13 * a21
            a1122_1221 = a11 * a22 - a12 * a21
            q1 = a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322
            q2 = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321
            q3 = a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221
            q4 = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221
            qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4
            if qsqr < evecprec:
                q1 = a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322
                q2 = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321
                q3 = a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221
                q4 = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221
                qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4
                if qsqr < evecprec:
                    rot = np.eye(3)
                    return (rmsd, rot, [q1, q2, q3, q4])
    normq = qsqr ** 0.5
    q1 /= normq
    q2 /= normq
    q3 /= normq
    q4 /= normq
    a2 = q1 * q1
    x2 = q2 * q2
    y2 = q3 * q3
    z2 = q4 * q4
    xy = q2 * q3
    az = q1 * q4
    zx = q4 * q2
    ay = q1 * q3
    yz = q3 * q4
    ax = q1 * q2
    rot = np.zeros((3, 3))
    rot[0][0] = a2 + x2 - y2 - z2
    rot[0][1] = 2 * (xy + az)
    rot[0][2] = 2 * (zx - ay)
    rot[1][0] = 2 * (xy - az)
    rot[1][1] = a2 - x2 + y2 - z2
    rot[1][2] = 2 * (yz + ax)
    rot[2][0] = 2 * (zx + ay)
    rot[2][1] = 2 * (yz - ax)
    rot[2][2] = a2 - x2 - y2 + z2
    return (rmsd, rot, (q1, q2, q3, q4))
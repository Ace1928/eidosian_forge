import numpy as np
def subdiagonalize(h_ii, s_ii, index_j):
    nb = h_ii.shape[0]
    nb_sub = len(index_j)
    h_sub_jj = get_subspace(h_ii, index_j)
    s_sub_jj = get_subspace(s_ii, index_j)
    e_j, v_jj = np.linalg.eig(np.linalg.solve(s_sub_jj, h_sub_jj))
    normalize(v_jj, s_sub_jj)
    permute_list = np.argsort(e_j.real)
    e_j = np.take(e_j, permute_list)
    v_jj = np.take(v_jj, permute_list, axis=1)
    c_ii = np.identity(nb, complex)
    for i in range(nb_sub):
        for j in range(nb_sub):
            c_ii[index_j[i], index_j[j]] = v_jj[i, j]
    h1_ii = rotate_matrix(h_ii, c_ii)
    s1_ii = rotate_matrix(s_ii, c_ii)
    return (h1_ii, s1_ii, c_ii, e_j)
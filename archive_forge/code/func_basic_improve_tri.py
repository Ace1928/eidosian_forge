def basic_improve_tri(manifold, tries=10):
    M = manifold.copy()
    for i in range(tries):
        if pos_tets(M):
            return M
        M.randomize()
    return manifold.copy()
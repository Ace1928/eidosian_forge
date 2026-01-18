def improved_triangulation(manifold, tries=10):
    M = basic_improve_tri(manifold, tries=tries)
    if pos_tets(M):
        return M
    curves = M.dual_curves()
    for i, c in zip(range(tries), curves):
        D = M.drill(c)
        D.dehn_fill((1, 0), M.num_cusps())
        F = D.filled_triangulation()
        if pos_tets(F):
            return F
    return manifold.copy()
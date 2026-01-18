import spherogram
def min_isosig_with_gluings(tangle, gluings, root=None):
    if root is not None:
        cs_name = cslabel(root)
    isosigs = []
    for i in range(tangle.boundary[0] + tangle.boundary[1]):
        rotated_tangle = tangle.reshape((tangle.boundary[0] + tangle.boundary[1], 0), i)
        if root is not None:
            rotated_root = crossing_strand_from_name(rotated_tangle, cs_name)
        else:
            rotated_root = None
        perm = range(tangle.boundary[0] + tangle.boundary[1])
        perm[tangle.boundary[0]:] = reversed(perm[tangle.boundary[0]:])
        perm = rotate_list(perm, i)
        rotated_gluings = []
        for g in gluings:
            new_g = [perm[g[0]], perm[g[1]]]
            new_g.sort()
            rotated_gluings.append(tuple(new_g))
        rotated_gluings.sort()
        isosigs.append(isosig_with_gluings(rotated_tangle, rotated_gluings, root=rotated_root))
    return min(isosigs)
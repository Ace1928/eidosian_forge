import spherogram
def isosig_with_gluings(tangle, gluings, root=None):
    return (isosig(tangle, root=root), tuple(gluings))
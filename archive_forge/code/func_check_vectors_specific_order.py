from pyomo.contrib.pynumero.dependencies import numpy as np, scipy
def check_vectors_specific_order(tst, v1, v1order, v2, v2order, v1_v2_map=None):
    tst.assertEqual(len(v1), len(v1order))
    tst.assertEqual(len(v2), len(v2order))
    tst.assertEqual(len(v1), len(v2))
    if v1_v2_map is None:
        v2map = {s: v2order.index(s) for s in v1order}
    else:
        v2map = {s: v2order.index(v1_v2_map[s]) for s in v1order}
    for i, s in enumerate(v1order):
        tst.assertAlmostEqual(v1[i], v2[v2map[s]], places=7)
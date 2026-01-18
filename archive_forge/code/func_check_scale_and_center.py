import pytest
import networkx as nx
def check_scale_and_center(self, pos, scale, center):
    center = np.array(center)
    low = center - scale
    hi = center + scale
    vpos = np.array(list(pos.values()))
    length = vpos.max(0) - vpos.min(0)
    assert (length <= 2 * scale).all()
    assert (vpos >= low).all()
    assert (vpos <= hi).all()
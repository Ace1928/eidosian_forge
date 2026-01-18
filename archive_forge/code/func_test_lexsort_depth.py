from pandas import MultiIndex
def test_lexsort_depth(self):
    levels = [[0, 1], [0, 1, 2]]
    index = MultiIndex(levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]], sortorder=2)
    assert index._lexsort_depth == 2
    index = MultiIndex(levels=levels, codes=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 2, 1]], sortorder=1)
    assert index._lexsort_depth == 1
    index = MultiIndex(levels=levels, codes=[[0, 0, 1, 0, 1, 1], [0, 1, 0, 2, 2, 1]], sortorder=0)
    assert index._lexsort_depth == 0
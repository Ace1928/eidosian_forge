from sympy.strategies.tree import treeapply, greedy, allresults, brute
from functools import partial, reduce
def test_allresults():
    assert set(allresults(inc)(3)) == {inc(3)}
    assert set(allresults([inc, dec])(3)) == {2, 4}
    assert set(allresults((inc, dec))(3)) == {3}
    assert set(allresults([inc, (dec, double)])(4)) == {5, 6}
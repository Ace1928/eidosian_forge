from sympy.strategies.branch.core import (
def test_chain():
    assert list(chain()(2)) == [2]
    assert list(chain(inc, inc)(2)) == [4]
    assert list(chain(branch5, inc)(4)) == [4]
    assert set(chain(branch5, inc)(5)) == {5, 7}
    assert list(chain(inc, branch5)(5)) == [7]
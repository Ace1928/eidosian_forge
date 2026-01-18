from .. import sage_helper
def rational_sqrt(x):
    """
        Given a nonnegative rational x, return a rational r which is close to
        sqrt(x) with the guarantee that r <= sqrt(x).
        """
    if x < 0:
        raise ValueError('negative input')
    elif x == 0:
        return QQ(0)
    for e in [50, 100, 500, 1000, 10000]:
        r = QQ(x).sqrt().bestappr(2 ** e)
        if r != 0:
            break
    assert r > 0
    if r ** 2 > x:
        r = x / r
    assert r ** 2 <= x
    return r
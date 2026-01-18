from mpmath import *
from mpmath import fp
def test_fp_expj():
    assert ae(fp.expj(0), 1.0 + 0j)
    assert ae(fp.expj(1), 0.5403023058681398 + 0.8414709848078965j)
    assert ae(fp.expj(2), -0.4161468365471424 + 0.9092974268256817j)
    assert ae(fp.expj(0.75), 0.7316888688738209 + 0.6816387600233341j)
    assert ae(fp.expj(2 + 3j), -0.02071873100224288 + 0.045271253156092976j)
    assert ae(fp.expjpi(0), 1.0 + 0j)
    assert ae(fp.expjpi(1), -1.0 + 0j)
    assert ae(fp.expjpi(2), 1.0 + 0j)
    assert ae(fp.expjpi(0.75), -0.7071067811865476 + 0.7071067811865476j)
    assert ae(fp.expjpi(2 + 3j), 8.06995175703046e-05 + 0j)
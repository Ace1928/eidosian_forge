from mpmath import *
def test_mpi_from_str():
    iv.dps = 15
    assert iv.convert('1.5 +- 0.5') == mpi(mpf('1.0'), mpf('2.0'))
    assert mpi(1, 2) in iv.convert('1.5 (33.33333333333333333333333333333%)')
    assert iv.convert('[1, 2]') == mpi(1, 2)
    assert iv.convert('1[2, 3]') == mpi(12, 13)
    assert iv.convert('1.[23,46]e-8') == mpi('1.23e-8', '1.46e-8')
    assert iv.convert('12[3.4,5.9]e4') == mpi('123.4e+4', '125.9e4')
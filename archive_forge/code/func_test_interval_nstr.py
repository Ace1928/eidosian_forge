from mpmath import *
def test_interval_nstr():
    iv.dps = n = 30
    x = mpi(1, 2)
    assert iv.nstr(x, n, mode='plusminus', error_dps=6) == '1.5 +- 0.5'
    assert iv.nstr(x, n, mode='plusminus', use_spaces=False, error_dps=6) == '1.5+-0.5'
    assert iv.nstr(x, n, mode='percent') == '1.5 (33.33%)'
    assert iv.nstr(x, n, mode='brackets', use_spaces=False) == '[1.0,2.0]'
    assert iv.nstr(x, n, mode='brackets', brackets=('<', '>')) == '<1.0, 2.0>'
    x = mpi('5.2582327113062393041', '5.2582327113062749951')
    assert iv.nstr(x, n, mode='diff') == '5.2582327113062[393041, 749951]'
    assert iv.nstr(iv.cos(mpi(1)), n, mode='diff', use_spaces=False) == '0.54030230586813971740093660744[2955,3053]'
    assert iv.nstr(mpi('1e123', '1e129'), n, mode='diff') == '[1.0e+123, 1.0e+129]'
    exp = iv.exp
    assert iv.nstr(iv.exp(mpi('5000.1')), n, mode='diff') == '3.2797365856787867069110487[0926, 1191]e+2171'
    iv.dps = 15
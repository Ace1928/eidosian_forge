import rpy2.rinterface as ri
def test_init_from_r():
    pairlist = ri.baseenv.find('pairlist')
    pl = pairlist(a=ri.StrSexpVector(['1']), b=ri.StrSexpVector(['3']))
    assert pl.typeof == ri.RTYPES.LISTSXP
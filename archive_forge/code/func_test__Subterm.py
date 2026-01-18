from __future__ import print_function
from patsy.util import no_pickling
def test__Subterm():
    s_ab = _expand_test_abbrevs([['a-', 'b-']])[0]
    s_abc = _expand_test_abbrevs([['a-', 'b-', 'c-']])[0]
    s_null = _expand_test_abbrevs([[]])[0]
    s_cd = _expand_test_abbrevs([['c-', 'd-']])[0]
    s_a = _expand_test_abbrevs([['a-']])[0]
    s_ap = _expand_test_abbrevs([['a+']])[0]
    s_abp = _expand_test_abbrevs([['a-', 'b+']])[0]
    for bad in (s_abc, s_null, s_cd, s_ap, s_abp):
        assert not s_ab.can_absorb(bad)
    assert s_ab.can_absorb(s_a)
    assert s_ab.absorb(s_a) == s_abp
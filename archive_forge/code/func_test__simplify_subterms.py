from __future__ import print_function
from patsy.util import no_pickling
def test__simplify_subterms():

    def t(given, expected):
        given = _expand_test_abbrevs(given)
        expected = _expand_test_abbrevs(expected)
        print('testing if:', given, '->', expected)
        _simplify_subterms(given)
        assert given == expected
    t([('a-',)], [('a-',)])
    t([(), ('a-',)], [('a+',)])
    t([(), ('a-',), ('b-',), ('a-', 'b-')], [('a+', 'b+')])
    t([(), ('a-',), ('a-', 'b-')], [('a+',), ('a-', 'b-')])
    t([('a-',), ('b-',), ('a-', 'b-')], [('b-',), ('a-', 'b+')])
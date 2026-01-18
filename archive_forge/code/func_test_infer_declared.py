from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_infer_declared(self):
    tests = [('no macaroons', [], {}, None), ('single macaroon with one declaration', [[checkers.Caveat(condition='declared foo bar')]], {'foo': 'bar'}, None), ('only one argument to declared', [[checkers.Caveat(condition='declared foo')]], {}, None), ('spaces in value', [[checkers.Caveat(condition='declared foo bar bloggs')]], {'foo': 'bar bloggs'}, None), ('attribute with declared prefix', [[checkers.Caveat(condition='declaredccf foo')]], {}, None), ('several macaroons with different declares', [[checkers.declared_caveat('a', 'aval'), checkers.declared_caveat('b', 'bval')], [checkers.declared_caveat('c', 'cval'), checkers.declared_caveat('d', 'dval')]], {'a': 'aval', 'b': 'bval', 'c': 'cval', 'd': 'dval'}, None), ('duplicate values', [[checkers.declared_caveat('a', 'aval'), checkers.declared_caveat('a', 'aval'), checkers.declared_caveat('b', 'bval')], [checkers.declared_caveat('a', 'aval'), checkers.declared_caveat('b', 'bval'), checkers.declared_caveat('c', 'cval'), checkers.declared_caveat('d', 'dval')]], {'a': 'aval', 'b': 'bval', 'c': 'cval', 'd': 'dval'}, None), ('conflicting values', [[checkers.declared_caveat('a', 'aval'), checkers.declared_caveat('a', 'conflict'), checkers.declared_caveat('b', 'bval')], [checkers.declared_caveat('a', 'conflict'), checkers.declared_caveat('b', 'another conflict'), checkers.declared_caveat('c', 'cval'), checkers.declared_caveat('d', 'dval')]], {'c': 'cval', 'd': 'dval'}, None), ('third party caveats ignored', [[checkers.Caveat(condition='declared a no conflict', location='location')], [checkers.declared_caveat('a', 'aval')]], {'a': 'aval'}, None), ('unparseable caveats ignored', [[checkers.Caveat(condition=' bad')], [checkers.declared_caveat('a', 'aval')]], {'a': 'aval'}, None), ('infer with namespace', [[checkers.declared_caveat('a', 'aval'), caveat_with_ns(checkers.declared_caveat('a', 'aval'), 'testns')]], {'a': 'aval'}, None)]
    for test in tests:
        uri_to_prefix = test[3]
        if uri_to_prefix is None:
            uri_to_prefix = {checkers.STD_NAMESPACE: ''}
        ns = checkers.Namespace(uri_to_prefix)
        print(test[0])
        ms = []
        for i, caveats in enumerate(test[1]):
            m = Macaroon(key=None, identifier=six.int2byte(i), location='', version=MACAROON_V2)
            for cav in caveats:
                cav = ns.resolve_caveat(cav)
                if cav.location == '':
                    m.add_first_party_caveat(cav.condition)
                else:
                    m.add_third_party_caveat(cav.location, None, cav.condition)
            ms.append(m)
        self.assertEqual(checkers.infer_declared(ms), test[2])
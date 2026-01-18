from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_checkers(self):
    tests = [('nothing in context, no extra checkers', [('something', 'caveat "something" not satisfied: caveat not recognized'), ('', 'cannot parse caveat "": empty caveat'), (' hello', 'cannot parse caveat " hello": caveat starts with space character')], None), ('one failed caveat', [('t:a aval', None), ('t:b bval', None), ('t:a wrong', 'caveat "t:a wrong" not satisfied: wrong arg')], None), ('time from clock', [(checkers.time_before_caveat(datetime.utcnow() + timedelta(0, 1)).condition, None), (checkers.time_before_caveat(NOW).condition, 'caveat "time-before 2006-01-02T15:04:05.000123Z" not satisfied: macaroon has expired'), (checkers.time_before_caveat(NOW - timedelta(0, 1)).condition, 'caveat "time-before 2006-01-02T15:04:04.000123Z" not satisfied: macaroon has expired'), ('time-before bad-date', 'caveat "time-before bad-date" not satisfied: cannot parse "bad-date" as RFC 3339'), (checkers.time_before_caveat(NOW).condition + ' ', 'caveat "time-before 2006-01-02T15:04:05.000123Z " not satisfied: cannot parse "2006-01-02T15:04:05.000123Z " as RFC 3339')], lambda x: checkers.context_with_clock(ctx, TestClock())), ('real time', [(checkers.time_before_caveat(datetime(year=2010, month=1, day=1)).condition, 'caveat "time-before 2010-01-01T00:00:00.000000Z" not satisfied: macaroon has expired'), (checkers.time_before_caveat(datetime(year=3000, month=1, day=1)).condition, None)], None), ('declared, no entries', [(checkers.declared_caveat('a', 'aval').condition, 'caveat "declared a aval" not satisfied: got a=null, expected "aval"'), (checkers.COND_DECLARED, 'caveat "declared" not satisfied: declared caveat has no value')], None), ('declared, some entries', [(checkers.declared_caveat('a', 'aval').condition, None), (checkers.declared_caveat('b', 'bval').condition, None), (checkers.declared_caveat('spc', ' a b').condition, None), (checkers.declared_caveat('a', 'bval').condition, 'caveat "declared a bval" not satisfied: got a="aval", expected "bval"'), (checkers.declared_caveat('a', ' aval').condition, 'caveat "declared a  aval" not satisfied: got a="aval", expected " aval"'), (checkers.declared_caveat('spc', 'a b').condition, 'caveat "declared spc a b" not satisfied: got spc=" a b", expected "a b"'), (checkers.declared_caveat('', 'a b').condition, 'caveat "error invalid caveat \'declared\' key """ not satisfied: bad caveat'), (checkers.declared_caveat('a b', 'a b').condition, 'caveat "error invalid caveat \'declared\' key "a b"" not satisfied: bad caveat')], lambda x: checkers.context_with_declared(x, {'a': 'aval', 'b': 'bval', 'spc': ' a b'}))]
    checker = checkers.Checker()
    checker.namespace().register('testns', 't')
    checker.register('a', 'testns', arg_checker(self, 't:a', 'aval'))
    checker.register('b', 'testns', arg_checker(self, 't:b', 'bval'))
    ctx = checkers.AuthContext()
    for test in tests:
        print(test[0])
        if test[2] is not None:
            ctx1 = test[2](ctx)
        else:
            ctx1 = ctx
        for check in test[1]:
            err = checker.check_first_party_caveat(ctx1, check[0])
            if check[1] is not None:
                self.assertEqual(err, check[1])
            else:
                self.assertIsNone(err)
from datetime import datetime, timedelta
from unittest import TestCase
import macaroonbakery.checkers as checkers
import six
from pymacaroons import MACAROON_V2, Macaroon
def test_checker_info(self):
    checker = checkers.Checker(include_std_checkers=False)
    checker.namespace().register('one', 't')
    checker.namespace().register('two', 't')
    checker.namespace().register('three', '')
    checker.namespace().register('four', 's')

    class Called(object):
        val = ''

    def register(name, ns):

        def func(ctx, cond, arg):
            Called.val = name + ' ' + ns
            return None
        checker.register(name, ns, func)
    register('x', 'one')
    register('y', 'one')
    register('z', 'two')
    register('a', 'two')
    register('something', 'three')
    register('other', 'three')
    register('xxx', 'four')
    expect = [checkers.CheckerInfo(ns='four', name='xxx', prefix='s'), checkers.CheckerInfo(ns='one', name='x', prefix='t'), checkers.CheckerInfo(ns='one', name='y', prefix='t'), checkers.CheckerInfo(ns='three', name='other', prefix=''), checkers.CheckerInfo(ns='three', name='something', prefix=''), checkers.CheckerInfo(ns='two', name='a', prefix='t'), checkers.CheckerInfo(ns='two', name='z', prefix='t')]
    infos = checker.info()
    self.assertEqual(len(infos), len(expect))
    new_infos = []
    for i, info in enumerate(infos):
        Called.val = ''
        info.check(None, '', '')
        self.assertEqual(Called.val, expect[i].name + ' ' + expect[i].ns)
        new_infos.append(checkers.CheckerInfo(ns=info.ns, name=info.name, prefix=info.prefix))
    self.assertEqual(new_infos, expect)
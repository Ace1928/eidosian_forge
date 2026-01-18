from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
def test_authorize_func(self):

    def f(ctx, identity, op):
        self.assertEqual(identity.id(), 'bob')
        if op.entity == 'a':
            return (False, None)
        elif op.entity == 'b':
            return (True, None)
        elif op.entity == 'c':
            return (True, [checkers.Caveat(location='somewhere', condition='c')])
        elif op.entity == 'd':
            return (True, [checkers.Caveat(location='somewhere', condition='d')])
        else:
            self.fail('unexpected entity: ' + op.Entity)
    ops = [bakery.Op('a', 'x'), bakery.Op('b', 'x'), bakery.Op('c', 'x'), bakery.Op('d', 'x')]
    allowed, caveats = bakery.AuthorizerFunc(f).authorize(checkers.AuthContext(), bakery.SimpleIdentity('bob'), ops)
    self.assertEqual(allowed, [False, True, True, True])
    self.assertEqual(caveats, [checkers.Caveat(location='somewhere', condition='c'), checkers.Caveat(location='somewhere', condition='d')])
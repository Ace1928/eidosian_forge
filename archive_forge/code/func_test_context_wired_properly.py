from unittest import TestCase
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
def test_context_wired_properly(self):
    ctx = checkers.AuthContext({'a': 'aval'})

    class Visited:
        in_f = False
        in_allow = False
        in_get_acl = False

    def f(ctx, identity, op):
        self.assertEqual(ctx.get('a'), 'aval')
        Visited.in_f = True
        return (False, None)
    bakery.AuthorizerFunc(f).authorize(ctx, bakery.SimpleIdentity('bob'), ['op1'])
    self.assertTrue(Visited.in_f)

    class TestIdentity(SimplestIdentity, bakery.ACLIdentity):

        def allow(other, ctx, acls):
            self.assertEqual(ctx.get('a'), 'aval')
            Visited.in_allow = True
            return False

    def get_acl(ctx, acl):
        self.assertEqual(ctx.get('a'), 'aval')
        Visited.in_get_acl = True
        return []
    bakery.ACLAuthorizer(allow_public=False, get_acl=get_acl).authorize(ctx, TestIdentity('bob'), ['op1'])
    self.assertTrue(Visited.in_get_acl)
    self.assertTrue(Visited.in_allow)
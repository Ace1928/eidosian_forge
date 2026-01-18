from datetime import datetime, timedelta
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
def test_checker():
    c = checkers.Checker()
    c.namespace().register('testns', '')
    c.register('str', 'testns', str_check)
    c.register('true', 'testns', true_check)
    return c
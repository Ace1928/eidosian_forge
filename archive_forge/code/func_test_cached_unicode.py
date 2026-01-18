from .. import cache_utf8
from . import TestCase
def test_cached_unicode(self):
    z = 'åzz'
    x = 'µyy' + z
    y = 'µyy' + z
    self.assertIsNot(x, y)
    xp = cache_utf8.get_cached_unicode(x)
    yp = cache_utf8.get_cached_unicode(y)
    self.assertIs(xp, x)
    self.assertIs(xp, yp)
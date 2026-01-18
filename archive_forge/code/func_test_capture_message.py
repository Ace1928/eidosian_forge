import warnings
import testtools
import fixtures
def test_capture_message(self):
    self.useFixture(fixtures.WarningsFilter())
    warnings.simplefilter('always')
    w = self.useFixture(fixtures.WarningsCapture())
    warnings.warn('hi', DeprecationWarning)
    self.assertEqual(1, len(w.captures))
    self.assertEqual('hi', str(w.captures[0].message))
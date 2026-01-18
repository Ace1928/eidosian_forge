import warnings
import testtools
import fixtures
def test_capture_reuse(self):
    self.useFixture(fixtures.WarningsFilter())
    warnings.simplefilter('always')
    w = fixtures.WarningsCapture()
    with w:
        warnings.warn('test', DeprecationWarning)
        self.assertEqual(1, len(w.captures))
    with w:
        self.assertEqual([], w.captures)
from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_switch_ascii(self):
    out, err = self.run_bzr_raw(['push', '--help'], encoding='ascii')
    self.assertContainsRe(out, b'zz\\?{{:See also:')
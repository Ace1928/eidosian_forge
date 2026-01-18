from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_switch_utf8(self):
    out, err = self.run_bzr_raw(['push', '--help'], encoding='utf-8')
    self.assertContainsRe(out, b'zz\xc3\xa5{{:See also:')
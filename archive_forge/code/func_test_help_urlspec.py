from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_urlspec(self):
    """Smoke test for 'brz help urlspec'"""
    out, err = self.run_bzr('help urlspec')
    self.assertContainsRe(out, 'bzr://')
    self.assertContainsRe(out, 'bzr\\+ssh://')
    self.assertContainsRe(out, 'file://')
    self.assertContainsRe(out, 'http://')
    self.assertContainsRe(out, 'https://')
    self.assertContainsRe(out, 'sftp://')
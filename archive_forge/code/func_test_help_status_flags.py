from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_status_flags(self):
    """Smoke test for 'brz help status-flags'"""
    out, err = self.run_bzr('help status-flags')
    from breezy.help_topics import _status_flags, help_as_plain_text
    expected = help_as_plain_text(_status_flags)
    self.assertEqual(expected, out)
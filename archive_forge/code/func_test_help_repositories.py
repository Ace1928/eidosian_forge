from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_repositories(self):
    """Smoke test for 'brz help repositories'"""
    out, err = self.run_bzr('help repositories')
    from breezy.help_topics import _repositories, help_as_plain_text
    expected = help_as_plain_text(_repositories)
    self.assertEqual(expected, out)
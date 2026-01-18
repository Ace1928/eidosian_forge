from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_help(self):
    help = self.run_bzr('help help')[0]
    qmark = self.run_bzr('? ?')[0]
    self.assertEqual(help, qmark)
    for line in help.split('\n'):
        if '--long' in line:
            self.assertContainsRe(line, 'Show help on all commands\\.')
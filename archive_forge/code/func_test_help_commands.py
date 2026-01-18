from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_commands(self):
    dash_help = self.run_bzr('--help commands')[0]
    commands = self.run_bzr('help commands')[0]
    hidden = self.run_bzr('help hidden-commands')[0]
    long_help = self.run_bzr('help --long')[0]
    qmark_long = self.run_bzr('? --long')[0]
    qmark_cmds = self.run_bzr('? commands')[0]
    self.assertEqual(dash_help, commands)
    self.assertEqual(dash_help, long_help)
    self.assertEqual(dash_help, qmark_long)
    self.assertEqual(dash_help, qmark_cmds)
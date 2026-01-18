from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
def test_help_width_zero(self):
    self.overrideEnv('BRZ_COLUMNS', '0')
    self.run_bzr('help commands')
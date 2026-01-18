import io
from .. import errors, i18n, tests, workingtree
def test_no_env_variables(self):
    self.overrideEnv('LANGUAGE', None)
    self.overrideEnv('LC_ALL', None)
    self.overrideEnv('LC_MESSAGES', None)
    self.overrideEnv('LANG', None)
    i18n.install()
    self.assertIsInstance(i18n._translations, i18n._gettext.NullTranslations)
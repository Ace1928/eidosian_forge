import io
from .. import errors, i18n, tests, workingtree
def test_disable_i18n(self):
    i18n.disable_i18n()
    i18n.install()
    self.assertIsInstance(i18n._translations, i18n._gettext.NullTranslations)
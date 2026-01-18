import io
from .. import errors, i18n, tests, workingtree
def test_custom_languages(self):
    i18n.install('nl:fy')
    self.assertIsInstance(i18n._translations, i18n._gettext.NullTranslations)
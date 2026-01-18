import gettext
import os
from oslo_i18n import _lazy
from oslo_i18n import _locale
from oslo_i18n import _message
@property
def plural_form(self):
    """The plural translation function.

        The returned function takes three values, the single form of
        the unicode string, the plural form of the unicode string,
        the count of items to be translated.

        .. versionadded:: 2.1.0

        """
    return self._make_plural_translation_func()
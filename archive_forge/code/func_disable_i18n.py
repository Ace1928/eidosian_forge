import gettext as _gettext
import os
import sys
def disable_i18n():
    """Do not allow i18n to be enabled.  Useful for third party users
    of breezy."""
    global _translations
    _translations = _gettext.NullTranslations()
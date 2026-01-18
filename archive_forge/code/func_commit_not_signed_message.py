import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def commit_not_signed_message(count):
    """returns message for number of commits"""
    return ngettext('{0} commit not signed', '{0} commits not signed', count[SIGNATURE_NOT_SIGNED]).format(count[SIGNATURE_NOT_SIGNED])
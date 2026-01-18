import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def expired_commit_message(count):
    """returns message for number of commits"""
    return ngettext('{0} commit with key now expired', '{0} commits with key now expired', count[SIGNATURE_EXPIRED]).format(count[SIGNATURE_EXPIRED])
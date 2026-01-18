import os as _os
import re as _re
import sys as _sys
import warnings
from gettext import gettext as _, ngettext
def parse_intermixed_args(self, args=None, namespace=None):
    args, argv = self.parse_known_intermixed_args(args, namespace)
    if argv:
        msg = _('unrecognized arguments: %s')
        self.error(msg % ' '.join(argv))
    return args
from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.compat import utf8, unichr, PY3, check_anchorname_char, nprint  # NOQA
@property
def scanner_processing_version(self):
    if hasattr(self.loader, 'typ'):
        return self.loader.resolver.processing_version
    return self.loader.processing_version
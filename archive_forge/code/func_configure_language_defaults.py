from __future__ import absolute_import
import os
from .. import Utils
def configure_language_defaults(self, source_extension):
    if source_extension == 'py':
        if self.compiler_directives.get('binding') is None:
            self.compiler_directives['binding'] = True
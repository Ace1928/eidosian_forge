import re
from distutils.extension import Extension as old_Extension
def has_cxx_sources(self):
    for source in self.sources:
        if cxx_ext_re(str(source)):
            return True
    return False
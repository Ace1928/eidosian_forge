from . import __version__
from .auxfuncs import (
from . import capi_maps
from . import func2subr
from .crackfortran import rmbadname
def dadd(line, s=doc):
    s[0] = '%s\n%s' % (s[0], line)
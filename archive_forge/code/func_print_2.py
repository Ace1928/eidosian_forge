import _sre
from . import _parser
from ._constants import *
from ._casefix import _EXTRA_CASES
def print_2(*args):
    print(end=' ' * (offset_width + 2 * level))
    print(*args)
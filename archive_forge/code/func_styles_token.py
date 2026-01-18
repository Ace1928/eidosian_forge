from pygments.token import Token, STANDARD_TYPES
from pygments.util import add_metaclass
def styles_token(cls, ttype):
    return ttype in cls._styles
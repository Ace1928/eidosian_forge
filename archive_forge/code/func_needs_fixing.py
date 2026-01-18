from lib2to3 import fixer_base
from lib2to3.fixer_util import token, String, Newline, Comma, Name
from libfuturize.fixer_util import indentation, suitify, DoubleStar
def needs_fixing(raw_params, kwargs_default=_kwargs_default_name):
    u"""
    Returns string with the name of the kwargs dict if the params after the first star need fixing
    Otherwise returns empty string
    """
    found_kwargs = False
    needs_fix = False
    for t in raw_params[2:]:
        if t.type == token.COMMA:
            continue
        elif t.type == token.NAME and (not found_kwargs):
            needs_fix = True
        elif t.type == token.NAME and found_kwargs:
            return t.value if needs_fix else u''
        elif t.type == token.DOUBLESTAR:
            found_kwargs = True
    else:
        return kwargs_default if needs_fix else u''
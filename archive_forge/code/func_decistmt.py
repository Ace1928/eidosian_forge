import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
@ipt.TokenInputTransformer.wrap
def decistmt(tokens):
    """Substitute Decimals for floats in a string of statements.

    Based on an example from the tokenize module docs.
    """
    result = []
    for toknum, tokval, _, _, _ in tokens:
        if toknum == tokenize.NUMBER and '.' in tokval:
            yield from [(tokenize.NAME, 'Decimal'), (tokenize.OP, '('), (tokenize.STRING, repr(tokval)), (tokenize.OP, ')')]
        else:
            yield (toknum, tokval)
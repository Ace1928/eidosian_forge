import ast
import string
def bytesjoin(iterable, joiner=b''):
    return tobytes(joiner).join((tobytes(item) for item in iterable))
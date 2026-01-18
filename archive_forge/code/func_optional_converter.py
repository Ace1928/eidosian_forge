import typing
from ._compat import _AnnotationExtractor
from ._make import NOTHING, Factory, pipe
def optional_converter(val):
    if val is None:
        return None
    return converter(val)
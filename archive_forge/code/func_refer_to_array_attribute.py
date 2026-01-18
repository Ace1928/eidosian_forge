from numpy.core.function_base import add_newdoc
from numpy.core.overrides import array_function_like_doc
def refer_to_array_attribute(attr, method=True):
    docstring = '\n    Scalar {} identical to the corresponding array attribute.\n\n    Please see `ndarray.{}`.\n    '
    return (attr, docstring.format('method' if method else 'attribute', attr))
import re
from collections import namedtuple
def py_type_name(type_name):
    """Get the Python type name for a given model type.

        >>> py_type_name('list')
        'list'
        >>> py_type_name('structure')
        'dict'

    :rtype: string
    """
    return {'blob': 'bytes', 'character': 'string', 'double': 'float', 'long': 'integer', 'map': 'dict', 'structure': 'dict', 'timestamp': 'datetime'}.get(type_name, type_name)
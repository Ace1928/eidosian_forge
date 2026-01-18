import ast
from .qt import ClassFlag, qt_class_flags
def format_member(attrib_node, qualifier_in='auto'):
    """Member access foo->member() is expressed as an attribute with
       further nested Attributes/Names as value"""
    n = attrib_node
    result = ''
    qualifier = qualifier_in
    if qualifier_in == 'auto':
        qualifier = '::' if n.attr[0:1].isupper() else '->'
    while isinstance(n, ast.Attribute):
        result = n.attr if not result else n.attr + qualifier + result
        n = n.value
    if isinstance(n, ast.Name) and n.id != 'self':
        if qualifier_in == 'auto' and n.id == 'Qt':
            qualifier = '::'
        result = n.id + qualifier + result
    return result
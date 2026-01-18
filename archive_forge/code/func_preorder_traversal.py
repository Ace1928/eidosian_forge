import sys
import pprint as _pprint_
from pyomo.common.collections import ComponentMap
import pyomo.core
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.kernel.base import (
def preorder_traversal(node, ctype=_no_ctype, active=True, descend=True):
    """
    A generator that yields each object in the storage tree
    (including the root object) using a preorder traversal.

    Args:
        node: The root object.
        ctype: Indicates the category of components to
            include. The default value indicates that all
            categories should be included.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend (bool, function): Controls if a container
            object should be descended into during the
            traversal. When a function is supplied, each
            container object will be passed into it and the
            return value will indicate if the traversal
            continues into children of the
            container. Default is True, which is equivalent
            to `lambda x: True`.

    Returns:
        iterator of objects in the storage tree, including
        the root object
    """
    assert active in (None, True)
    if active is not None and (not node.active):
        return
    ctype = _convert_ctype.get(ctype, ctype)
    descend = _convert_descend_into(descend)
    if ctype is _no_ctype or node.ctype is ctype or node.ctype._is_heterogeneous_container:
        yield node
    if not node._is_container or not descend(node):
        return
    for child in node.children():
        child_ctype = child.ctype
        if not child._is_container:
            if active is None or child.active:
                if ctype is _no_ctype or child_ctype is ctype:
                    yield child
        elif child._is_heterogeneous_container:
            for obj in preorder_traversal(child, ctype=ctype, active=active, descend=descend):
                yield obj
        elif child_ctype._is_heterogeneous_container:

            def descend_(obj_):
                if obj_._is_heterogeneous_container:
                    return False
                else:
                    return descend(obj_)
            for obj in preorder_traversal(child, active=active, descend=descend_):
                if not obj._is_heterogeneous_container:
                    yield obj
                else:
                    for item in preorder_traversal(obj, ctype=ctype, active=active, descend=descend):
                        yield item
        elif ctype is _no_ctype or child_ctype is ctype:
            for obj in preorder_traversal(child, active=active, descend=descend):
                yield obj
from pyomo.core.kernel.base import (
def heterogeneous_containers(node, ctype=_no_ctype, active=True, descend_into=True):
    """
    A generator that yields all heterogeneous containers
    included in an object storage tree, including the root
    object. Heterogeneous containers are categorized objects
    with a category type different from their children.

    Args:
        node: The root object.
        ctype: Indicates the category of objects to
            include. The default value indicates that
            all categories should be included.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend_into (bool, function): Indicates whether or
            not to descend into a heterogeneous
            container. Default is True, which is equivalent
            to `lambda x: True`, meaning all heterogeneous
            containers will be descended into.

    Returns:
        iterator of heterogeneous containers in the storage
        tree, include the root object.
    """
    assert active in (None, True)
    if active is not None and (not node.active):
        return
    if not node.ctype._is_heterogeneous_container:
        return
    if not node._is_heterogeneous_container:
        assert node._is_container
        for obj in node.components(active=active):
            assert obj._is_heterogeneous_container
            yield from heterogeneous_containers(obj, ctype=ctype, active=active, descend_into=descend_into)
        return
    ctype = _convert_ctype.get(ctype, ctype)
    assert ctype is _no_ctype or ctype._is_heterogeneous_container
    descend_into = _convert_descend_into(descend_into)
    if ctype is _no_ctype or node.ctype is ctype:
        yield node
    if not descend_into(node):
        return
    for child_ctype in node.child_ctypes():
        if not child_ctype._is_heterogeneous_container:
            continue
        for child in node.children(ctype=child_ctype):
            assert child._is_container
            if active is not None and (not child.active):
                continue
            yield from heterogeneous_containers(child, ctype=ctype, active=active, descend_into=descend_into)
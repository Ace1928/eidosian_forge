from pyomo.core.kernel.base import (
def collect_ctypes(self, active=True, descend_into=True):
    """Returns the set of object category types that can
        be found under this container.

        Args:
            active (:const:`True`/:const:`None`): Controls
                whether or not to filter the iteration to
                include only the active part of the storage
                tree. The default is :const:`True`. Setting
                this keyword to :const:`None` causes the
                active status of objects to be ignored.
            descend_into (bool, function): Indicates whether
                or not to descend into a heterogeneous
                container. Default is True, which is
                equivalent to `lambda x: True`, meaning all
                heterogeneous containers will be descended
                into.

        Returns:
            A set of object category types
        """
    assert active in (None, True)
    ctypes = set()
    if active is not None and (not self.active):
        return ctypes
    descend_into = _convert_descend_into(descend_into)
    for child_ctype in self.child_ctypes():
        for obj in self.components(ctype=child_ctype, active=active, descend_into=_convert_descend_into._false):
            ctypes.add(child_ctype)
            break
    if descend_into is _convert_descend_into._false:
        return ctypes
    for child_ctype in tuple(ctypes):
        if child_ctype._is_heterogeneous_container:
            for obj in self.components(ctype=child_ctype, active=active, descend_into=_convert_descend_into._false):
                assert obj._is_heterogeneous_container
                if descend_into(obj):
                    ctypes.update(obj.collect_ctypes(active=active, descend_into=descend_into))
    return ctypes
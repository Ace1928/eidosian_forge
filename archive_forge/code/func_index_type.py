from __future__ import absolute_import
import math, sys
def index_type(base_type, item):
    """
    Support array type creation by slicing, e.g. double[:, :] specifies
    a 2D strided array of doubles. The syntax is the same as for
    Cython memoryviews.
    """

    class InvalidTypeSpecification(Exception):
        pass

    def verify_slice(s):
        if s.start or s.stop or s.step not in (None, 1):
            raise InvalidTypeSpecification('Only a step of 1 may be provided to indicate C or Fortran contiguity')
    if isinstance(item, tuple):
        step_idx = None
        for idx, s in enumerate(item):
            verify_slice(s)
            if s.step and (step_idx or idx not in (0, len(item) - 1)):
                raise InvalidTypeSpecification('Step may only be provided once, and only in the first or last dimension.')
            if s.step == 1:
                step_idx = idx
        return _ArrayType(base_type, len(item), is_c_contig=step_idx == len(item) - 1, is_f_contig=step_idx == 0)
    elif isinstance(item, slice):
        verify_slice(item)
        return _ArrayType(base_type, 1, is_c_contig=bool(item.step))
    else:
        assert int(item) == item
        return array(base_type, item)
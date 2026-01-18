import sys, types
from .lock import allocate_lock
from .error import CDefError
from . import model
def new_allocator(self, alloc=None, free=None, should_clear_after_alloc=True):
    """Return a new allocator, i.e. a function that behaves like ffi.new()
        but uses the provided low-level 'alloc' and 'free' functions.

        'alloc' is called with the size as argument.  If it returns NULL, a
        MemoryError is raised.  'free' is called with the result of 'alloc'
        as argument.  Both can be either Python function or directly C
        functions.  If 'free' is None, then no free function is called.
        If both 'alloc' and 'free' are None, the default is used.

        If 'should_clear_after_alloc' is set to False, then the memory
        returned by 'alloc' is assumed to be already cleared (or you are
        fine with garbage); otherwise CFFI will clear it.
        """
    compiled_ffi = self._backend.FFI()
    allocator = compiled_ffi.new_allocator(alloc, free, should_clear_after_alloc)

    def allocate(cdecl, init=None):
        if isinstance(cdecl, basestring):
            cdecl = self._typeof(cdecl)
        return allocator(cdecl, init)
    return allocate
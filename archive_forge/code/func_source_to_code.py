import _imp
import _io
import sys
import _warnings
import marshal
def source_to_code(self, data, path, *, _optimize=-1):
    """Return the code object compiled from source.

        The 'data' argument can be any object type that compile() supports.
        """
    return _bootstrap._call_with_frames_removed(compile, data, path, 'exec', dont_inherit=True, optimize=_optimize)
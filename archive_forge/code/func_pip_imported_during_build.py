import sys
import os
@classmethod
def pip_imported_during_build(cls):
    """
        Detect if pip is being imported in a build script. Ref #2355.
        """
    import traceback
    return any((cls.frame_file_is_setup(frame) for frame, line in traceback.walk_stack(None)))
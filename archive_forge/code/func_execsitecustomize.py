import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def execsitecustomize():
    """Run custom site specific code, if available."""
    try:
        try:
            import sitecustomize
        except ImportError as exc:
            if exc.name == 'sitecustomize':
                pass
            else:
                raise
    except Exception as err:
        if sys.flags.verbose:
            sys.excepthook(*sys.exc_info())
        else:
            sys.stderr.write('Error in sitecustomize; set PYTHONVERBOSE for traceback:\n%s: %s\n' % (err.__class__.__name__, err))
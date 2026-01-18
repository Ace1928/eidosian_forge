import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def setcopyright():
    """Set 'copyright' and 'credits' in builtins"""
    builtins.copyright = _sitebuiltins._Printer('copyright', sys.copyright)
    if sys.platform[:4] == 'java':
        builtins.credits = _sitebuiltins._Printer('credits', 'Jython is maintained by the Jython developers (www.jython.org).')
    else:
        builtins.credits = _sitebuiltins._Printer('credits', '    Thanks to CWI, CNRI, BeOpen.com, Zope Corporation and a cast of thousands\n    for supporting Python development.  See www.python.org for more information.')
    files, dirs = ([], [])
    here = getattr(sys, '_stdlib_dir', None)
    if not here and hasattr(os, '__file__'):
        here = os.path.dirname(os.__file__)
    if here:
        files.extend(['LICENSE.txt', 'LICENSE'])
        dirs.extend([os.path.join(here, os.pardir), here, os.curdir])
    builtins.license = _sitebuiltins._Printer('license', 'See https://www.python.org/psf/license/', files, dirs)
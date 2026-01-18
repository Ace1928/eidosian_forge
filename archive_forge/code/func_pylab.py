from traitlets.config.application import Application
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from warnings import warn
from IPython.core.pylabtools import backends
@skip_doctest
@line_magic
@magic_arguments.magic_arguments()
@magic_arguments.argument('--no-import-all', action='store_true', default=None, help='Prevent IPython from performing ``import *`` into the interactive namespace.\n\n        You can govern the default behavior of this flag with the\n        InteractiveShellApp.pylab_import_all configurable.\n        ')
@magic_gui_arg
def pylab(self, line=''):
    """Load numpy and matplotlib to work interactively.

        This function lets you activate pylab (matplotlib, numpy and
        interactive support) at any point during an IPython session.

        %pylab makes the following imports::

            import numpy
            import matplotlib
            from matplotlib import pylab, mlab, pyplot
            np = numpy
            plt = pyplot

            from IPython.display import display
            from IPython.core.pylabtools import figsize, getfigs

            from pylab import *
            from numpy import *

        If you pass `--no-import-all`, the last two `*` imports will be excluded.

        See the %matplotlib magic for more details about activating matplotlib
        without affecting the interactive namespace.
        """
    args = magic_arguments.parse_argstring(self.pylab, line)
    if args.no_import_all is None:
        if Application.initialized():
            app = Application.instance()
            try:
                import_all = app.pylab_import_all
            except AttributeError:
                import_all = True
        else:
            import_all = True
    else:
        import_all = not args.no_import_all
    gui, backend, clobbered = self.shell.enable_pylab(args.gui, import_all=import_all)
    self._show_matplotlib_backend(args.gui, backend)
    print('%pylab is deprecated, use %matplotlib inline and import the required libraries.')
    print('Populating the interactive namespace from numpy and matplotlib')
    if clobbered:
        warn('pylab import has clobbered these variables: %s' % clobbered + '\n`%matplotlib` prevents importing * from pylab and numpy')
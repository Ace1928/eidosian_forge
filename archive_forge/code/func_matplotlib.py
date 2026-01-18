from traitlets.config.application import Application
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from warnings import warn
from IPython.core.pylabtools import backends
@skip_doctest
@line_magic
@magic_arguments.magic_arguments()
@magic_arguments.argument('-l', '--list', action='store_true', help='Show available matplotlib backends')
@magic_gui_arg
def matplotlib(self, line=''):
    """Set up matplotlib to work interactively.

        This function lets you activate matplotlib interactive support
        at any point during an IPython session. It does not import anything
        into the interactive namespace.

        If you are using the inline matplotlib backend in the IPython Notebook
        you can set which figure formats are enabled using the following::

            In [1]: from matplotlib_inline.backend_inline import set_matplotlib_formats

            In [2]: set_matplotlib_formats('pdf', 'svg')

        The default for inline figures sets `bbox_inches` to 'tight'. This can
        cause discrepancies between the displayed image and the identical
        image created using `savefig`. This behavior can be disabled using the
        `%config` magic::

            In [3]: %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}

        In addition, see the docstrings of
        `matplotlib_inline.backend_inline.set_matplotlib_formats` and
        `matplotlib_inline.backend_inline.set_matplotlib_close` for more information on
        changing additional behaviors of the inline backend.

        Examples
        --------
        To enable the inline backend for usage with the IPython Notebook::

            In [1]: %matplotlib inline

        In this case, where the matplotlib default is TkAgg::

            In [2]: %matplotlib
            Using matplotlib backend: TkAgg

        But you can explicitly request a different GUI backend::

            In [3]: %matplotlib qt

        You can list the available backends using the -l/--list option::

           In [4]: %matplotlib --list
           Available matplotlib backends: ['osx', 'qt4', 'qt5', 'gtk3', 'gtk4', 'notebook', 'wx', 'qt', 'nbagg',
           'gtk', 'tk', 'inline']
        """
    args = magic_arguments.parse_argstring(self.matplotlib, line)
    if args.list:
        backends_list = list(backends.keys())
        print('Available matplotlib backends: %s' % backends_list)
    else:
        gui, backend = self.shell.enable_matplotlib(args.gui.lower() if isinstance(args.gui, str) else args.gui)
        self._show_matplotlib_backend(args.gui, backend)
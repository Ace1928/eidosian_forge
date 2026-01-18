from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
@magic_arguments.magic_arguments()
@magic_arguments.argument('filename', type=str, help='Notebook name or filename')
@line_magic
def notebook(self, s):
    """Export and convert IPython notebooks.

        This function can export the current IPython history to a notebook file.
        For example, to export the history to "foo.ipynb" do "%notebook foo.ipynb".
        """
    args = magic_arguments.parse_argstring(self.notebook, s)
    outfname = os.path.expanduser(args.filename)
    from nbformat import write, v4
    cells = []
    hist = list(self.shell.history_manager.get_range())
    if len(hist) <= 1:
        raise ValueError('History is empty, cannot export')
    for session, execution_count, source in hist[:-1]:
        cells.append(v4.new_code_cell(execution_count=execution_count, source=source))
    nb = v4.new_notebook(cells=cells)
    with io.open(outfname, 'w', encoding='utf-8') as f:
        write(nb, f, version=4)
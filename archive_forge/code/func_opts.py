import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@line_cell_magic
def opts(self, line='', cell=None):
    """
        The opts line/cell magic with tab-completion.

        %%opts [ [path] [normalization] [plotting options] [style options]]+

        path:             A dotted type.group.label specification
                          (e.g. Image.Grayscale.Photo)

        normalization:    List of normalization options delimited by braces.
                          One of | -axiswise | -framewise | +axiswise | +framewise |
                          E.g. { +axiswise +framewise }

        plotting options: List of plotting option keywords delimited by
                          square brackets. E.g. [show_title=False]

        style options:    List of style option keywords delimited by
                          parentheses. E.g. (lw=10 marker='+')

        Note that commas between keywords are optional (not
        recommended) and that keywords must end in '=' without a
        separating space.

        More information may be found in the class docstring of
        util.parser.OptsSpec.
        """
    line, cell = self._partition_lines(line, cell)
    try:
        spec = OptsSpec.parse(line, ns=self.shell.user_ns)
    except SyntaxError:
        display(HTML('<b>Invalid syntax</b>: Consult <tt>%%opts?</tt> for more information.'))
        return
    available_elements = set()
    for backend in Store.loaded_backends():
        available_elements |= set(Store.options(backend).children)
    spec_elements = {k.split('.')[0] for k in spec.keys()}
    unknown_elements = spec_elements - available_elements
    if unknown_elements:
        msg = '<b>WARNING:</b> Unknown elements {unknown} not registered with any of the loaded backends.'
        display(HTML(msg.format(unknown=', '.join(unknown_elements))))
    if cell:
        self.register_custom_spec(spec)
        self.shell.run_cell(cell, store_history=STORE_HISTORY)
    else:
        errmsg = StoreOptions.validation_error_message(spec)
        if errmsg:
            OptsMagic.error_message = None
            sys.stderr.write(errmsg)
            if self.strict:
                display(HTML('Options specification will not be applied.'))
                return
        with options_policy(skip_invalid=True, warn_on_skip=False):
            StoreOptions.apply_customizations(spec, Store.options())
    OptsMagic.error_message = None
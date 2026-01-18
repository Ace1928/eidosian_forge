import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
def cell_runner(cell, renderer):
    self.shell.run_cell(cell, store_history=STORE_HISTORY)
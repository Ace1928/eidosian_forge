import math
import re
import shutil
import rich
import rich.console
import rich.markup
import rich.table
import tree
from keras.src import backend
from keras.src.utils import dtype_utils
from keras.src.utils import io_utils
def highlight_symbol(x):
    """Themes keras symbols in a summary using rich markup."""
    return f'[color(33)]{x}[/]'
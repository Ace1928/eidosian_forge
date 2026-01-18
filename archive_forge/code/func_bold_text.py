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
def bold_text(x, color=None):
    """Bolds text using rich markup."""
    if color:
        return f'[bold][color({color})]{x}[/][/]'
    return f'[bold]{x}[/]'
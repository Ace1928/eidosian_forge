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
def highlight_number(x):
    """Themes numbers in a summary using rich markup.

    We use a separate color for `None`s, e.g. in a layer shape.
    """
    if x is None:
        return f'[color(45)]{x}[/]'
    else:
        return f'[color(34)]{x}[/]'
import os
import sys
import re
from collections.abc import Iterator
from warnings import warn
from looseversion import LooseVersion
import numpy as np
import textwrap
def container_to_string(cont):
    """Convert a container to a command line string.

    Elements of the container are joined with a space between them,
    suitable for a command line parameter.

    If the container `cont` is only a sequence, like a string and not a
    container, it is returned unmodified.

    Parameters
    ----------
    cont : container
       A container object like a list, tuple, dict, or a set.

    Returns
    -------
    cont_str : string
        Container elements joined into a string.

    """
    if hasattr(cont, '__iter__') and (not isinstance(cont, str)):
        cont = ' '.join(cont)
    return str(cont)
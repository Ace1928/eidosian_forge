import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
def render_pep440_old(pieces):
    """TAG[.postDISTANCE[.dev0]] .

    The ".dev0" means dirty.

    Exceptions:
    1: no tags. 0.postDISTANCE[.dev0]
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += f'0.post{pieces['distance']}'
            if pieces['dirty']:
                rendered += '.dev0'
    else:
        rendered = f'0.post{pieces['distance']}'
        if pieces['dirty']:
            rendered += '.dev0'
    return rendered
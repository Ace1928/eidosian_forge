import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
def render_pep440(pieces):
    """Build up version string, with post-release "local version identifier".

    Our goal: TAG[+DISTANCE.gHEX[.dirty]] . Note that if you
    get a tagged build and then dirty it, you'll get TAG+0.gHEX.dirty

    Exceptions:
    1: no tags. git_describe was just HEX. 0+untagged.DISTANCE.gHEX[.dirty]
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        if pieces['distance'] or pieces['dirty']:
            rendered += plus_or_dot(pieces)
            rendered += f'{pieces['distance']}.g{pieces['short']}'
            if pieces['dirty']:
                rendered += '.dirty'
    else:
        rendered = f'0+untagged.{pieces['distance']}.g{pieces['short']}'
        if pieces['dirty']:
            rendered += '.dirty'
    return rendered
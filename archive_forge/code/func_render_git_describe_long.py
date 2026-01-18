import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
def render_git_describe_long(pieces):
    """TAG-DISTANCE-gHEX[-dirty].

    Like 'git describe --tags --dirty --always -long'.
    The distance/hash is unconditional.

    Exceptions:
    1: no tags. HEX[-dirty]  (note: no 'g' prefix)
    """
    if pieces['closest-tag']:
        rendered = pieces['closest-tag']
        rendered += f'-{pieces['distance']}-g{pieces['short']}'
    else:
        rendered = pieces['short']
    if pieces['dirty']:
        rendered += '-dirty'
    return rendered
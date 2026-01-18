from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
@classmethod
def process_normalization(cls, parse_group):
    """
        Given a normalization parse group (i.e. the contents of the
        braces), validate the option list and compute the appropriate
        integer value for the normalization plotting option.
        """
    if 'norm_options' not in parse_group:
        return None
    opts = parse_group['norm_options'][0].asList()
    if opts == []:
        return None
    options = ['+framewise', '-framewise', '+axiswise', '-axiswise']
    for normopt in options:
        if opts.count(normopt) > 1:
            raise SyntaxError('Normalization specification must not contain repeated %r' % normopt)
    if not all((opt in options for opt in opts)):
        raise SyntaxError(f'Normalization option not one of {', '.join(options)}')
    excluded = [('+framewise', '-framewise'), ('+axiswise', '-axiswise')]
    for pair in excluded:
        if all((exclude in opts for exclude in pair)):
            raise SyntaxError(f'Normalization specification cannot contain both {pair[0]} and {pair[1]}')
    if len(opts) == 1 and opts[0].endswith('framewise'):
        axiswise = False
        framewise = True if '+framewise' in opts else False
    elif len(opts) == 1 and opts[0].endswith('axiswise'):
        framewise = False
        axiswise = True if '+axiswise' in opts else False
    else:
        axiswise = True if '+axiswise' in opts else False
        framewise = True if '+framewise' in opts else False
    return dict(axiswise=axiswise, framewise=framewise)
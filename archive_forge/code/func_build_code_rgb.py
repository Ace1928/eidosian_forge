import logging
from logging import Formatter
from typing import NamedTuple, Tuple, Union
def build_code_rgb(rgb: Tuple[int, int, int], rgb_bg: Union[None, Tuple[int, int, int]]=None):
    """
    Utility function to generate the appropriate ANSI RGB codes for a given set of foreground (font) and background colors.
    """
    output = _ANSI_CODES['begin']
    output += _ANSI_CODES['foreground_rgb']
    output += ';'.join([str(i) for i in rgb])
    output += 'm'
    if rgb_bg:
        output += _ANSI_CODES['begin']
        output += _ANSI_CODES['background_rgb']
        output += ';'.join([str(i) for i in rgb_bg])
        output += 'm'
    return output
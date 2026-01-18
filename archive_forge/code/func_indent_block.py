import os
import textwrap
from passlib.utils.compat import irange
def indent_block(block, padding):
    """ident block of text"""
    lines = block.split('\n')
    return '\n'.join((padding + line if line else '' for line in lines))
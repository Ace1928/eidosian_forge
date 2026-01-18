from __future__ import print_function
import os
def cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    """Print colorize text.

    It accepts arguments of print function.
    """
    print(colored(text, color, on_color, attrs), **kwargs)
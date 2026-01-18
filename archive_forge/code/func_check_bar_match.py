import math
import warnings
import matplotlib.dates
def check_bar_match(old_bar, new_bar):
    """Check if two bars belong in the same collection (bar chart).

    Positional arguments:
    old_bar -- a previously sorted bar dictionary.
    new_bar -- a new bar dictionary that needs to be sorted.

    """
    tests = []
    tests += (new_bar['orientation'] == old_bar['orientation'],)
    tests += (new_bar['facecolor'] == old_bar['facecolor'],)
    if new_bar['orientation'] == 'v':
        new_width = new_bar['x1'] - new_bar['x0']
        old_width = old_bar['x1'] - old_bar['x0']
        tests += (new_width - old_width < 1e-06,)
        tests += (new_bar['y0'] == old_bar['y0'],)
    elif new_bar['orientation'] == 'h':
        new_height = new_bar['y1'] - new_bar['y0']
        old_height = old_bar['y1'] - old_bar['y0']
        tests += (new_height - old_height < 1e-06,)
        tests += (new_bar['x0'] == old_bar['x0'],)
    if all(tests):
        return True
    else:
        return False
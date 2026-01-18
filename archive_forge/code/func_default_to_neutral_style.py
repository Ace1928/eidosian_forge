import itertools
import functools
import importlib.util
def default_to_neutral_style(fn):
    """Wrap a function or method to use the neutral style by default."""

    @functools.wraps(fn)
    def wrapper(*args, style='neutral', **kwargs):
        import matplotlib.pyplot as plt
        if style == 'neutral':
            style = NEUTRAL_STYLE
        elif not style:
            style = {}
        with plt.style.context(style):
            return fn(*args, **kwargs)
    return wrapper
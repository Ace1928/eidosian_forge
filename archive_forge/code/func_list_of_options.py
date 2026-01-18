from collections.abc import Sequence
from plotly import exceptions
from plotly.colors import (
def list_of_options(iterable, conj='and', period=True):
    """
    Returns an English listing of objects seperated by commas ','

    For example, ['foo', 'bar', 'baz'] becomes 'foo, bar and baz'
    if the conjunction 'and' is selected.
    """
    if len(iterable) < 2:
        raise exceptions.PlotlyError('Your list or tuple must contain at least 2 items.')
    template = (len(iterable) - 2) * '{}, ' + '{} ' + conj + ' {}' + period * '.'
    return template.format(*iterable)
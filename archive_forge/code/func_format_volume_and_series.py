from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_volume_and_series(self, e, as_sentence=True):
    volume_and_series = optional[words[together['Volume' if as_sentence else 'volume', field('volume')], optional[words['of', field('series')]]]]
    number_and_series = optional[words[join(sep=Symbol('nbsp'))['Number' if as_sentence else 'number', field('number')], optional[words['in', field('series')]]]]
    series = optional_field('series')
    result = first_of[volume_and_series, number_and_series, series]
    if as_sentence:
        return sentence(capfirst=True)[result]
    else:
        return result
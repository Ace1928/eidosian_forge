import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def paint_statusbar(rows, columns, msg, config):
    func = func_for_letter(config.color_scheme['main'])
    return fsarray([func(msg.ljust(columns))[:columns]])
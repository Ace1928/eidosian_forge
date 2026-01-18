import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def paginate(rows, matches, current, words_wide):
    if current not in matches:
        current = matches[0]
    per_page = rows * words_wide
    current_page = matches.index(current) // per_page
    return matches[per_page * current_page:per_page * (current_page + 1)]
import logging
import itertools
from curtsies import fsarray, fmtstr, FSArray
from curtsies.formatstring import linesplit
from curtsies.fmtfuncs import bold
from .parse import func_for_letter
def matches_lines(rows, columns, matches, current, config, match_format):
    highlight_color = func_for_letter(config.color_scheme['operator'].lower())
    if not matches:
        return []
    color = func_for_letter(config.color_scheme['main'])
    max_match_width = max((len(m) for m in matches))
    words_wide = max(1, (columns - 1) // (max_match_width + 1))
    matches = [match_format(m) for m in matches]
    if current:
        current = match_format(current)
    matches = paginate(rows, matches, current, words_wide)
    result = [fmtstr(' ').join((color(m.ljust(max_match_width)) if m != current else highlight_color(m.ljust(max_match_width)) for m in matches[i:i + words_wide])) for i in range(0, len(matches), words_wide)]
    logger.debug('match: %r' % current)
    logger.debug('matches_lines: %r' % result)
    return result
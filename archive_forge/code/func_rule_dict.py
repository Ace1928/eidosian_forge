import io
import math
import os
import typing
import weakref
def rule_dict(item):
    """Make a Python dict from a PDF page label rule.

    Args:
        item -- a tuple (pno, rule) with the start page number and the rule
                string like <</S/D...>>.
    Returns:
        A dict like
        {'startpage': int, 'prefix': str, 'style': str, 'firstpagenum': int}.
    """
    pno, rule = item
    rule = rule[2:-2].split('/')[1:]
    d = {'startpage': pno, 'prefix': '', 'firstpagenum': 1}
    skip = False
    for i, item in enumerate(rule):
        if skip:
            skip = False
            continue
        if item == 'S':
            d['style'] = rule[i + 1]
            skip = True
            continue
        if item.startswith('P'):
            x = item[1:].replace('(', '').replace(')', '')
            d['prefix'] = x
            continue
        if item.startswith('St'):
            x = int(item[2:])
            d['firstpagenum'] = x
    return d
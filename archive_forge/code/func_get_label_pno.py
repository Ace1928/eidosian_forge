import io
import math
import os
import typing
import weakref
def get_label_pno(pgNo, labels):
    """Return the label for this page number.

    Args:
        pgNo: page number, 0-based.
        labels: result of doc._get_page_labels().
    Returns:
        The label (str) of the page number. Errors return an empty string.
    """
    item = [x for x in labels if x[0] <= pgNo][-1]
    rule = rule_dict(item)
    prefix = rule.get('prefix', '')
    style = rule.get('style', '')
    pagenumber = pgNo - rule['startpage'] + rule['firstpagenum']
    return construct_label(style, prefix, pagenumber)
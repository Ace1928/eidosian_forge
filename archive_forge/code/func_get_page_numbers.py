import io
import math
import os
import typing
import weakref
def get_page_numbers(doc, label, only_one=False):
    """Return a list of page numbers with the given label.

    Args:
        doc: PDF document object (resp. 'self').
        label: (str) label.
        only_one: (bool) stop searching after first hit.
    Returns:
        List of page numbers having this label.
    """
    numbers = []
    if not label:
        return numbers
    labels = doc._get_page_labels()
    if labels == []:
        return numbers
    for i in range(doc.page_count):
        plabel = get_label_pno(i, labels)
        if plabel == label:
            numbers.append(i)
            if only_one:
                break
    return numbers
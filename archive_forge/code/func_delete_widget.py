import io
import math
import os
import typing
import weakref
def delete_widget(page: fitz.Page, widget: fitz.Widget) -> fitz.Widget:
    """Delete widget from page and return the next one."""
    fitz.CheckParent(page)
    annot = getattr(widget, '_annot', None)
    if annot is None:
        raise ValueError('bad type: widget')
    nextwidget = widget.next
    page.delete_annot(annot)
    widget._annot.parent = None
    keylist = list(widget.__dict__.keys())
    for key in keylist:
        del widget.__dict__[key]
    return nextwidget
import lxml.etree as ET
from functools import partial
def add_text(elem, item):
    try:
        last_child = elem[-1]
    except IndexError:
        elem.text = (elem.text or '') + item
    else:
        last_child.tail = (last_child.tail or '') + item
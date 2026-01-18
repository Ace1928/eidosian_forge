import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
from bs4.builder import (
def linkage_validator(self, el, _recursive_call=False):
    """Ensure proper linkage throughout the document."""
    descendant = None
    if el.parent is None:
        assert el.previous_element is None, 'Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}'.format(el, el.previous_element, None)
        assert el.previous_sibling is None, 'Bad previous_sibling\nNODE: {}\nPREV: {}\nEXPECTED: {}'.format(el, el.previous_sibling, None)
        assert el.next_sibling is None, 'Bad next_sibling\nNODE: {}\nNEXT: {}\nEXPECTED: {}'.format(el, el.next_sibling, None)
    idx = 0
    child = None
    last_child = None
    last_idx = len(el.contents) - 1
    for child in el.contents:
        descendant = None
        if idx == 0:
            if el.parent is not None:
                assert el.next_element is child, 'Bad next_element\nNODE: {}\nNEXT: {}\nEXPECTED: {}'.format(el, el.next_element, child)
                assert child.previous_element is el, 'Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}'.format(child, child.previous_element, el)
                assert child.previous_sibling is None, 'Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED: {}'.format(child, child.previous_sibling, None)
        else:
            assert child.previous_sibling is el.contents[idx - 1], 'Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED {}'.format(child, child.previous_sibling, el.contents[idx - 1])
            assert el.contents[idx - 1].next_sibling is child, 'Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(el.contents[idx - 1], el.contents[idx - 1].next_sibling, child)
            if last_child is not None:
                assert child.previous_element is last_child, 'Bad previous_element\nNODE: {}\nPREV {}\nEXPECTED {}\nCONTENTS {}'.format(child, child.previous_element, last_child, child.parent.contents)
                assert last_child.next_element is child, 'Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(last_child, last_child.next_element, child)
        if isinstance(child, Tag) and child.contents:
            descendant = self.linkage_validator(child, True)
            assert descendant.next_sibling is None, 'Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(descendant, descendant.next_sibling, None)
        if descendant is not None:
            last_child = descendant
        else:
            last_child = child
        if idx == last_idx:
            assert child.next_sibling is None, 'Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(child, child.next_sibling, None)
        idx += 1
    child = descendant if descendant is not None else child
    if child is None:
        child = el
    if not _recursive_call and child is not None:
        target = el
        while True:
            if target is None:
                assert child.next_element is None, 'Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(child, child.next_element, None)
                break
            elif target.next_sibling is not None:
                assert child.next_element is target.next_sibling, 'Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}'.format(child, child.next_element, target.next_sibling)
                break
            target = target.parent
        return None
    else:
        return child
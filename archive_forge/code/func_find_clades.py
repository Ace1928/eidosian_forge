import collections
import copy
import itertools
import random
import re
import warnings
def find_clades(self, target=None, terminal=None, order='preorder', **kwargs):
    """Find each clade containing a matching element.

        That is, find each element as with find_elements(), but return the
        corresponding clade object. (This is usually what you want.)

        :returns: an iterable through all matching objects, searching
            depth-first (preorder) by default.

        """

    def match_attrs(elem):
        orig_clades = elem.__dict__.pop('clades')
        found = elem.find_any(target, **kwargs)
        elem.clades = orig_clades
        return found is not None
    if terminal is None:
        is_matching_elem = match_attrs
    else:

        def is_matching_elem(elem):
            return elem.is_terminal() == terminal and match_attrs(elem)
    return self._filter_search(is_matching_elem, order, False)
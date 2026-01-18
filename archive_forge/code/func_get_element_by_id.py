import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def get_element_by_id(self, id, *default):
    """
        Get the first element in a document with the given id.  If none is
        found, return the default argument if provided or raise KeyError
        otherwise.

        Note that there can be more than one element with the same id,
        and this isn't uncommon in HTML documents found in the wild.
        Browsers return only the first match, and this function does
        the same.
        """
    try:
        return _id_xpath(self, id=id)[0]
    except IndexError:
        if default:
            return default[0]
        else:
            raise KeyError(id)
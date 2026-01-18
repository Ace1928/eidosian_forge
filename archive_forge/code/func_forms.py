import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@property
def forms(self):
    """
        Return a list of all the forms
        """
    return _forms_xpath(self)
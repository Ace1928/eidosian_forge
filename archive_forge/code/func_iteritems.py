import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
def iteritems(self):
    """ iterator through the sections names and content
        in the documentation Page. (deprecated, use items()) """
    warnings.warn('Use the method items().', DeprecationWarning)
    return self.sections.items()
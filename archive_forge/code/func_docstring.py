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
def docstring(package: Package, alias: str, sections: typing.Tuple[str, ...]=('\\usage', '\\arguments')) -> str:
    """Fetch the R documentation for an alias in a package."""
    if not isinstance(package, Package):
        package = Package(package)
    page = package.fetch(alias)
    return page.to_docstring(sections)
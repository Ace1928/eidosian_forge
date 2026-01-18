import sys
from .constants import FIELD_TYPE
from .err import (
from .times import (
from . import connections  # noqa: E402
def install_as_MySQLdb():
    """
    After this function is called, any application that imports MySQLdb
    will unwittingly actually use pymysql.
    """
    sys.modules['MySQLdb'] = sys.modules['pymysql']
from .release import version_info
from . import _mysql
from ._mysql import (
from MySQLdb.constants import FIELD_TYPE
from MySQLdb.times import (
def test_DBAPISet_set_equality_membership():
    assert FIELD_TYPE.VAR_STRING == STRING
import dill
from enum import EnumMeta
import sys
from collections import namedtuple
def test_enummeta():
    from http import HTTPStatus
    import enum
    assert dill.copy(HTTPStatus.OK) is HTTPStatus.OK
    assert dill.copy(enum.EnumMeta) is enum.EnumMeta
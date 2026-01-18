import pytest
from rpy2 import robjects
def test_isclass(set_class_A):
    ainstance = robjects.r('new("A", a=1, b="c")')
    assert not ainstance.isclass('B')
    assert ainstance.isclass('A')
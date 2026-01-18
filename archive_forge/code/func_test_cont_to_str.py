import os
from shutil import rmtree
import pytest
from nipype.utils.misc import (
def test_cont_to_str():
    x = ['a', 'b']
    assert container_to_string(x) == 'a b'
    x = tuple(x)
    assert container_to_string(x) == 'a b'
    x = set(x)
    y = container_to_string(x)
    assert y == 'a b' or y == 'b a'
    x = dict(a='a', b='b')
    y = container_to_string(x)
    assert y == 'a b' or y == 'b a'
    assert container_to_string('foobar') == 'foobar'
    assert container_to_string(123) == '123'
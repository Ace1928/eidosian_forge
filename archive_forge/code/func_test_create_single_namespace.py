import pytest
import sys
from pathlib import Path
import catalogue
def test_create_single_namespace():
    test_registry = catalogue.create('test')
    assert catalogue.REGISTRY == {}

    @test_registry.register('a')
    def a():
        pass

    def b():
        pass
    test_registry.register('b', func=b)
    items = test_registry.get_all()
    assert len(items) == 2
    assert items['a'] == a
    assert items['b'] == b
    assert catalogue.check_exists('test', 'a')
    assert catalogue.check_exists('test', 'b')
    assert catalogue._get(('test', 'a')) == a
    assert catalogue._get(('test', 'b')) == b
    with pytest.raises(TypeError):

        @test_registry.register('x', 'y')
        def x():
            pass
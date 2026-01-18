import pytest
import sys
from pathlib import Path
import catalogue
def test_create_multi_namespace():
    test_registry = catalogue.create('x', 'y')

    @test_registry.register('z')
    def z():
        pass
    items = test_registry.get_all()
    assert len(items) == 1
    assert items['z'] == z
    assert catalogue.check_exists('x', 'y', 'z')
    assert catalogue._get(('x', 'y', 'z')) == z
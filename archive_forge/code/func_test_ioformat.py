import pytest
from ase.io.formats import ioformats
@pytest.mark.parametrize('name', ioformats)
def test_ioformat(name):
    ioformat = ioformats[name]
    print(name)
    print('=' * len(name))
    print(ioformat.full_description())
    print()
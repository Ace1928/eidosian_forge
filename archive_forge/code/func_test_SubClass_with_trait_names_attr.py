from IPython.utils.dir2 import dir2
import pytest
def test_SubClass_with_trait_names_attr():

    class SubClass(Base):
        y = 2
        trait_names = 44
    res = dir2(SubClass())
    assert 'trait_names' in res
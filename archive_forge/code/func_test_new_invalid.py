import pytest
import rpy2.rinterface as rinterface
def test_new_invalid():
    x = 1
    with pytest.raises(TypeError):
        rinterface.SexpSymbol(x)
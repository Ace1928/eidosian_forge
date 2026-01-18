import pytest
from .. import utils
import rpy2.rinterface as rinterface
@pytest.mark.skip(reason='WIP')
def test_from_pyobject_invalid_protected():
    pyobject = 'ahaha'
    with pytest.raises(TypeError):
        rinterface.SexpExtPtr.from_pyobject(pyobject, protected=True)
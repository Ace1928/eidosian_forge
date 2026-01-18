import pytest
from .. import utils
import rpy2.rinterface as rinterface
def test_getitem_invalid():
    env = rinterface.baseenv['new.env']()
    with pytest.raises(TypeError):
        env[None]
    with pytest.raises(ValueError):
        env['']
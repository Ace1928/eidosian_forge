import pytest
import rpy2.robjects as robjects
import rpy2.robjects.language as lg
from rpy2 import rinterface
from rpy2.rinterface_lib import embedded
def test_LangVector_from_string_invalid():
    with pytest.raises(embedded.RRuntimeError):
        lang_obj = lg.LangVector.from_string(1)
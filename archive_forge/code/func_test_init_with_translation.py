import pytest
import rpy2.robjects as robjects
import array
def test_init_with_translation():
    ri_f = rinterface.baseenv.find('rank')
    ro_f = SignatureTranslatedFunction(ri_f, init_prm_translate={'foo_bar': 'na.last'})
    assert identical(ri_f, ro_f)[0] is True
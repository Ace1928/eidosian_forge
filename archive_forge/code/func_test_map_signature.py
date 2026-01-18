import pytest
import inspect
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('r_code,parameter_names,r_ellipsis', (('function(x, y=FALSE, z="abc") TRUE', ('x', 'y', 'z'), None), ('function(x, y=FALSE, z="abc") {TRUE}', ('x', 'y', 'z'), None), ('function(x, ..., y=FALSE, z="abc") TRUE', ('x', '___', 'y', 'z'), 1)))
def test_map_signature(r_code, parameter_names, r_ellipsis):
    r_func = robjects.r(r_code)
    stf = robjects.functions.SignatureTranslatedFunction(r_func)
    signature, r_ellipsis = robjects.functions.map_signature(r_func)
    assert tuple(signature.parameters.keys()) == parameter_names
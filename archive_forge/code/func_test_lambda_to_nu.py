from numpy.testing import assert_equal, assert_allclose
import scipy.constants as sc
def test_lambda_to_nu():
    assert_equal(sc.lambda2nu([sc.speed_of_light, 1]), [1, sc.speed_of_light])
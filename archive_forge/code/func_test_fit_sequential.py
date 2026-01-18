import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_fit_sequential():
    np.random.seed(435265)
    X = np.random.normal(size=(50, 3))
    y = np.random.randint(0, 2, size=50)
    mod = DistributedModel(1, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 1), parallel_method='sequential', fit_kwds={'alpha': 0.5})
    assert_allclose(fit.params, np.array([-0.191606, -0.012565, -0.351398]), atol=1e-06, rtol=0)
    mod = DistributedModel(2, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 2), parallel_method='sequential', fit_kwds={'alpha': 0.5})
    assert_allclose(fit.params, np.array([-0.157416, -0.029643, -0.471653]), atol=1e-06, rtol=0)
    mod = DistributedModel(3, model_class=OLS)
    fit = mod.fit(_data_gen(y, X, 3), parallel_method='sequential', fit_kwds={'alpha': 0.5})
    assert_allclose(fit.params, np.array([-0.124891, -0.050934, -0.403354]), atol=1e-06, rtol=0)
    mod = DistributedModel(1, model_class=GLM, init_kwds={'family': Binomial()})
    fit = mod.fit(_data_gen(y, X, 1), parallel_method='sequential', fit_kwds={'alpha': 0.5})
    assert_allclose(fit.params, np.array([-0.164515, -0.412854, -0.223955]), atol=1e-06, rtol=0)
    mod = DistributedModel(2, model_class=GLM, init_kwds={'family': Binomial()})
    fit = mod.fit(_data_gen(y, X, 2), parallel_method='sequential', fit_kwds={'alpha': 0.5})
    assert_allclose(fit.params, np.array([-0.142513, -0.360324, -0.295485]), atol=1e-06, rtol=0)
    mod = DistributedModel(3, model_class=GLM, init_kwds={'family': Binomial()})
    fit = mod.fit(_data_gen(y, X, 3), parallel_method='sequential', fit_kwds={'alpha': 0.5})
    assert_allclose(fit.params, np.array([-0.110487, -0.306431, -0.243921]), atol=1e-06, rtol=0)
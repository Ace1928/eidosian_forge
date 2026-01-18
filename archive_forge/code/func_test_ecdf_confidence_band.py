import pytest
import numpy as np
import scipy.stats
from ...stats.ecdf_utils import (
@pytest.mark.parametrize('prob', [0.8, 0.9])
@pytest.mark.parametrize('dist, rvs', [(scipy.stats.norm(3, 10), scipy.stats.norm(3, 10).rvs), (scipy.stats.norm(3, 10), None), (scipy.stats.poisson(100), scipy.stats.poisson(100).rvs)], ids=['continuous', 'continuous default rvs', 'discrete'])
@pytest.mark.parametrize('ndraws', [10000])
@pytest.mark.parametrize('method', ['pointwise', 'simulated'])
def test_ecdf_confidence_band(dist, rvs, prob, ndraws, method, num_trials=1000, seed=57):
    """Test test_ecdf_confidence_band."""
    eval_points = np.linspace(*dist.interval(0.99), 10)
    cdf_at_eval_points = dist.cdf(eval_points)
    random_state = np.random.default_rng(seed)
    ecdf_lower, ecdf_upper = ecdf_confidence_band(ndraws, eval_points, cdf_at_eval_points, prob=prob, rvs=rvs, random_state=random_state, method=method)
    if method == 'pointwise':
        ecdf_lower_pointwise, ecdf_upper_pointwise = _get_pointwise_confidence_band(prob, ndraws, cdf_at_eval_points)
        assert np.array_equal(ecdf_lower, ecdf_lower_pointwise)
        assert np.array_equal(ecdf_upper, ecdf_upper_pointwise)
        return
    assert np.all(ecdf_lower >= 0)
    assert np.all(ecdf_upper <= 1)
    assert np.all(ecdf_lower <= ecdf_upper)
    in_envelope = []
    random_state = np.random.default_rng(seed)
    for _ in range(num_trials):
        ecdf = _simulate_ecdf(ndraws, eval_points, dist.rvs, random_state=random_state)
        in_envelope.append(np.all(ecdf_lower <= ecdf) & np.all(ecdf < ecdf_upper))
    asymptotic_dist = scipy.stats.norm(np.mean(in_envelope), scipy.stats.sem(in_envelope))
    prob_lower, prob_upper = asymptotic_dist.interval(0.999)
    assert prob_lower <= prob <= prob_upper
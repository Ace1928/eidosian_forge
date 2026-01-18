import random
import unittest
from cirq_ft.linalg.lcu_util import (
def assertPreprocess(self, lcu_coefs, epsilon):
    alternates, keep_numers, mu = preprocess_lcu_coefficients_for_reversible_sampling(lcu_coefs, epsilon)
    n = len(lcu_coefs)
    keep_denom = 2 ** mu
    self.assertEqual(len(alternates), n)
    self.assertEqual(len(keep_numers), n)
    self.assertTrue(all((0 <= e < keep_denom for e in keep_numers)))
    out_distribution = [1 / n * numer / keep_denom for numer in keep_numers]
    for i in range(n):
        switch_probability = 1 - keep_numers[i] / keep_denom
        out_distribution[alternates[i]] += 1 / n * switch_probability
    total = sum(lcu_coefs)
    for i in range(n):
        self.assertAlmostEqual(out_distribution[i], lcu_coefs[i] / total, delta=epsilon)
    return (alternates, keep_numers, keep_denom)
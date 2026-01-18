import random
import unittest
from cirq_ft.linalg.lcu_util import (
def assertGetDiscretizedDistribution(self, probabilities, epsilon):
    total_probability = sum(probabilities)
    numers, denom, mu = _discretize_probability_distribution(probabilities, epsilon)
    self.assertEqual(sum(numers), denom)
    self.assertEqual(len(numers), len(probabilities))
    self.assertEqual(len(probabilities) * 2 ** mu, denom)
    for i in range(len(numers)):
        self.assertAlmostEqual(numers[i] / denom, probabilities[i] / total_probability, delta=epsilon)
    return (numers, denom)
import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def random_equal_permutations(n_perms, n_items, prob):
    indices_to_permute = [i for i in range(n_items) if random.random() <= prob]
    permuted_indices = random.sample(indices_to_permute, len(indices_to_permute))
    base_permutation = dict(zip(indices_to_permute, permuted_indices))
    fixed_indices = [i for i in range(n_items) if i not in base_permutation]
    permutations = []
    for _ in range(n_perms):
        permutation = base_permutation.copy()
        permutation.update({i: i for i in fixed_indices if random.random() <= prob})
        permutations.append(permutation)
    return permutations
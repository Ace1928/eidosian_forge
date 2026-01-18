import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def random_permutation_equality_groups(n_groups, n_perms_per_group, n_items, prob):
    fingerprints = set()
    for _ in range(n_groups):
        perms = random_equal_permutations(n_perms_per_group, n_items, prob)
        perm = perms[0]
        fingerprint = tuple((perm.get(i, i) for i in range(n_items)))
        if fingerprint not in fingerprints:
            yield perms
            fingerprints.add(fingerprint)
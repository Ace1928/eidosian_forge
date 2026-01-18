import cirq
import pytest
def test_init_product_of_sums():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.ProductOfSums([1]), cirq.ProductOfSums(((1,),)))
    eq.add_equality_group(cirq.ProductOfSums([0]), cirq.ProductOfSums(((0,),)))
    eq.add_equality_group(cirq.ProductOfSums([[0, 1, 2]]), cirq.ProductOfSums(((0, 1, 2),)))
    eq.add_equality_group(cirq.ProductOfSums([0, 1, 2]), cirq.ProductOfSums([[0, 0], [1, 1], [2, 2]]), cirq.ProductOfSums([[0], [1], [2]]))
    eq.add_equality_group([0, 0, 1, 1, 2, 2])
    eq.add_equality_group([2, 0, 1])
    eq.add_equality_group(cirq.ProductOfSums([(0, 1), (1, 2)]), cirq.ProductOfSums([(1, 0), (2, 1)]), cirq.ProductOfSums([[0, 1], (2, 1)]))
    eq.add_equality_group([(1, 2), (0, 1)])
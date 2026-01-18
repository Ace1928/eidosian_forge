import cirq
import pytest
def test_equality_across_types():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.SumOfProducts([[1]]), cirq.ProductOfSums([1]), cirq.SumOfProducts(((1,),)), cirq.ProductOfSums(((1,),)))
    eq.add_equality_group(((1,),), tuple(cirq.ProductOfSums([1])), tuple(cirq.SumOfProducts([[1]])))
    eq.add_equality_group(cirq.SumOfProducts([[0], [1], [2]]), cirq.ProductOfSums([[0, 1, 2]]))
    eq.add_equality_group(cirq.ProductOfSums([0, 1, 2]), cirq.SumOfProducts([[0, 1, 2]]))
    eq.add_equality_group(((0,), (1,), (2,)), tuple(cirq.SumOfProducts([[0], [1], [2]])), tuple(cirq.ProductOfSums([0, 1, 2])))
    eq.add_equality_group(((0, 1, 2),), tuple(cirq.ProductOfSums([[0, 1, 2]])), tuple(cirq.SumOfProducts([[0, 1, 2]])))
    eq.add_equality_group(cirq.ProductOfSums([(0, 1), (1, 2), 1]), cirq.SumOfProducts([(0, 1, 1), (0, 2, 1), (1, 1, 1), (1, 2, 1)]))
    eq.add_equality_group(cirq.SumOfProducts([(0, 1), (1, 0)]))
    eq.add_equality_group(cirq.ProductOfSums([(0, 1), (1, 0)]))
import cirq
import pytest
def test_and_operation():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.ProductOfSums([0]) & cirq.ProductOfSums([1]), cirq.ProductOfSums([0]) & cirq.SumOfProducts([[1]]), cirq.SumOfProducts([[0]]) & cirq.ProductOfSums([1]), cirq.ProductOfSums([0, 1]), cirq.SumOfProducts([[0, 1]]))
    eq.add_equality_group(cirq.ProductOfSums([1]) & cirq.SumOfProducts([[0]]), cirq.ProductOfSums([1, 0]))
    eq.add_equality_group(cirq.ProductOfSums([[0, 1]]) & cirq.ProductOfSums([1]), cirq.SumOfProducts([[0], [1]]) & cirq.ProductOfSums([1]), cirq.ProductOfSums([[0, 1], [1]]), cirq.SumOfProducts([[0, 1], [1, 1]]))
    eq.add_equality_group(cirq.ProductOfSums([0, 1]) & cirq.ProductOfSums([0]), cirq.ProductOfSums([0]) & cirq.ProductOfSums([1, 0]))
    eq.add_equality_group(cirq.SumOfProducts([(0, 0), (1, 1)]) & cirq.ProductOfSums([0, 1]), cirq.SumOfProducts([(0, 0), (1, 1)]) & cirq.ProductOfSums([0]) & cirq.ProductOfSums([1]), cirq.SumOfProducts([(0, 0, 0, 1), (1, 1, 0, 1)]))
    eq.add_equality_group(cirq.SumOfProducts([(0, 1), (1, 0)]) & cirq.SumOfProducts([(0, 0), (0, 1), (1, 0)]), cirq.SumOfProducts([(0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0)]))
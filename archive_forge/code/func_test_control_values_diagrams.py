import cirq
import pytest
def test_control_values_diagrams():
    q = cirq.LineQubit.range(3)
    ccx = cirq.X(q[0]).controlled_by(*q[1:])
    ccx_sop = cirq.X(q[0]).controlled_by(*q[1:], control_values=cirq.SumOfProducts([[1, 1]]))
    cirq.testing.assert_has_diagram(cirq.Circuit(ccx, ccx_sop), '\n0: ───X───X───\n      │   │\n1: ───@───@───\n      │   │\n2: ───@───@───\n')
    c0c1x = cirq.X(q[0]).controlled_by(*q[1:], control_values=[0, 1])
    c0c1x_sop = cirq.X(q[0]).controlled_by(*q[1:], control_values=cirq.SumOfProducts([[0, 1]]))
    cirq.testing.assert_has_diagram(cirq.Circuit(c0c1x, c0c1x_sop), '\n0: ───X─────X─────\n      │     │\n1: ───(0)───(0)───\n      │     │\n2: ───@─────@─────\n')
    c01c2x = cirq.X(q[0]).controlled_by(*q[1:], control_values=[[0, 1], 1])
    c01c2x_sop = cirq.X(q[0]).controlled_by(*q[1:], control_values=cirq.SumOfProducts([[0, 1], [1, 1]]))
    cirq.testing.assert_has_diagram(cirq.Circuit(c01c2x, c01c2x_sop), '\n0: ───X───────X───────\n      │       │\n1: ───(0,1)───@(01)───\n      │       │\n2: ───@───────@(11)───\n')
    xor_sop = cirq.X(q[0]).controlled_by(*q[1:], control_values=cirq.SumOfProducts([[0, 1], [1, 0]]))
    xor_named_sop = cirq.X(q[0]).controlled_by(*q[1:], control_values=cirq.SumOfProducts([[0, 1], [1, 0]], name='xor'))
    cirq.testing.assert_has_diagram(cirq.Circuit(xor_sop, xor_named_sop), '\n0: ───X───────X────────\n      │       │\n1: ───@(01)───@────────\n      │       │\n2: ───@(10)───@(xor)───\n        ')
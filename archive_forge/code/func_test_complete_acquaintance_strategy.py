from string import ascii_lowercase as alphabet
import pytest
import cirq
import cirq.contrib.acquaintance as cca
def test_complete_acquaintance_strategy():
    qubits = [cirq.NamedQubit(s) for s in alphabet]
    with pytest.raises(ValueError):
        _ = cca.complete_acquaintance_strategy(qubits, -1)
    empty_strategy = cca.complete_acquaintance_strategy(qubits)
    assert empty_strategy._moments == []
    trivial_strategy = cca.complete_acquaintance_strategy(qubits[:4], 1)
    actual_text_diagram = trivial_strategy.to_text_diagram().strip()
    expected_text_diagram = '\na: ───█───\n\nb: ───█───\n\nc: ───█───\n\nd: ───█───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(trivial_strategy) == 1
    quadratic_strategy = cca.complete_acquaintance_strategy(qubits[:8], 2)
    actual_text_diagram = quadratic_strategy.to_text_diagram().strip()
    expected_text_diagram = '\na: ───×(0,0)───\n      │\nb: ───×(1,0)───\n      │\nc: ───×(2,0)───\n      │\nd: ───×(3,0)───\n      │\ne: ───×(4,0)───\n      │\nf: ───×(5,0)───\n      │\ng: ───×(6,0)───\n      │\nh: ───×(7,0)───\n    '.strip()
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(quadratic_strategy) == 2
    is_shift_or_lin_perm = lambda op: isinstance(op.gate, (cca.CircularShiftGate, cca.LinearPermutationGate))
    quadratic_strategy = cirq.expand_composite(quadratic_strategy, no_decomp=is_shift_or_lin_perm)
    actual_text_diagram = quadratic_strategy.to_text_diagram(transpose=True).strip()
    expected_text_diagram = '\n'.join(('a   b   c   d   e   f   g   h        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '█───█   █───█   █───█   █───█        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   █───█   █───█   █───█   │        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '█───█   █───█   █───█   █───█        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   █───█   █───█   █───█   │        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '█───█   █───█   █───█   █───█        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   █───█   █───█   █───█   │        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '█───█   █───█   █───█   █───█        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲      '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   █───█   █───█   █───█   │        '.strip(), '│   │   │   │   │   │   │   │        '.strip(), '│   ╲0╱─╱1╲ ╲0╱─╱1╲ ╲0╱─╱1╲ │        '.strip(), '│   │   │   │   │   │   │   │        '.strip()))
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(quadratic_strategy) == 2
    cubic_strategy = cca.complete_acquaintance_strategy(qubits[:4], 3)
    actual_text_diagram = cubic_strategy.to_text_diagram(transpose=True).strip()
    expected_text_diagram = '\na      b      c      d\n│      │      │      │\n×(0,0)─×(0,1)─×(1,0)─×(1,1)\n│      │      │      │\n╱1╲────╲0╱    ╱1╲────╲0╱\n│      │      │      │\n×(0,0)─×(1,0)─×(1,1)─×(2,0)\n│      │      │      │\n│      ╲0╱────╱1╲    │\n│      │      │      │\n×(0,0)─×(0,1)─×(1,0)─×(1,1)\n│      │      │      │\n╱1╲────╲0╱    ╱1╲────╲0╱\n│      │      │      │\n×(0,0)─×(1,0)─×(1,1)─×(2,0)\n│      │      │      │\n│      ╲0╱────╱1╲    │\n│      │      │      │\n    '.strip()
    assert actual_text_diagram == expected_text_diagram
    assert cca.get_acquaintance_size(cubic_strategy) == 3
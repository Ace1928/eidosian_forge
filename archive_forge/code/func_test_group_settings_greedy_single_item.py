import cirq
def test_group_settings_greedy_single_item():
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    term = cirq.X(q0) * cirq.X(q1)
    settings = list(cirq.work.observables_to_settings([term], qubits))
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 1
    assert list(grouped_settings.keys())[0] == settings[0]
    assert list(grouped_settings.values())[0][0] == settings[0]
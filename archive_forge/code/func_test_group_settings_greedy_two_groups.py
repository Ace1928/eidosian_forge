import cirq
def test_group_settings_greedy_two_groups():
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    terms = [cirq.X(q0) * cirq.X(q1), cirq.Y(q0) * cirq.Y(q1)]
    settings = list(cirq.work.observables_to_settings(terms, qubits))
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 2
    group_max_obs_should_be = terms.copy()
    group_max_settings_should_be = list(cirq.work.observables_to_settings(group_max_obs_should_be, qubits))
    assert set(grouped_settings.keys()) == set(group_max_settings_should_be)
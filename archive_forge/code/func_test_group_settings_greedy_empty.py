import cirq
def test_group_settings_greedy_empty():
    assert cirq.work.group_settings_greedy([]) == dict()
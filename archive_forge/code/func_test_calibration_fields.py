import datetime
import cirq
import cirq_ionq as ionq
def test_calibration_fields():
    calibration_dict = {'connectivity': [[0, 1], [0, 2], [1, 2]], 'target': 'ionq.qpu', 'qubits': 3, 'fidelity': {'1q': {'mean': 0.999}, '2q': {'mean': 0.999}}, 'timing': {'1q': 1.1e-05, '2q': 0.00021, 'readout': 0.000175, 'reset': 3.5e-05}, 'date': '2020-08-07T12:47:22.337Z'}
    cal = ionq.Calibration(calibration_dict=calibration_dict)
    assert cal.connectivity() == {(cirq.LineQubit(0), cirq.LineQubit(1)), (cirq.LineQubit(1), cirq.LineQubit(0)), (cirq.LineQubit(0), cirq.LineQubit(2)), (cirq.LineQubit(2), cirq.LineQubit(0)), (cirq.LineQubit(1), cirq.LineQubit(2)), (cirq.LineQubit(2), cirq.LineQubit(1))}
    assert cal.target() == 'ionq.qpu'
    assert cal.num_qubits() == 3
    assert cal.fidelities() == {'1q': {'mean': 0.999}, '2q': {'mean': 0.999}}
    assert cal.timings() == {'1q': 1.1e-05, '2q': 0.00021, 'readout': 0.000175, 'reset': 3.5e-05}
    assert cal.calibration_time(tz=datetime.timezone.utc) == datetime.datetime(year=2020, month=8, day=7, hour=12, minute=47, second=22, microsecond=337000, tzinfo=datetime.timezone.utc)
    time_minus_one = cal.calibration_time(tz=datetime.timezone(datetime.timedelta(hours=-2)))
    assert time_minus_one == datetime.datetime(year=2020, month=8, day=7, hour=12, minute=47, second=22, microsecond=337000, tzinfo=datetime.timezone.utc)
    assert time_minus_one.tzinfo == datetime.timezone(datetime.timedelta(hours=-2))
from collections import namedtuple
from typing import Dict, List, Optional, Union
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.duration import convert_durations_to_dt
from qiskit.circuit.measure import Measure
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.pulse import Schedule
from qiskit.pulse import instructions as pulse_inst
from qiskit.pulse.channels import AcquireChannel, MemorySlot, DriveChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.macros import measure
from qiskit.scheduler.config import ScheduleConfig
from qiskit.providers import BackendV1, BackendV2
def get_measure_schedule(qubit_mem_slots: Dict[int, int]) -> CircuitPulseDef:
    """Create a schedule to measure the qubits queued for measuring."""
    sched = Schedule()
    acquire_excludes = {}
    if Measure().name in circuit.calibrations.keys():
        qubits = tuple(sorted(qubit_mem_slots.keys()))
        params = ()
        for qubit in qubits:
            try:
                meas_q = circuit.calibrations[Measure().name][(qubit,), params]
                meas_q = target_qobj_transform(meas_q)
                acquire_q = meas_q.filter(channels=[AcquireChannel(qubit)])
                mem_slot_index = [chan.index for chan in acquire_q.channels if isinstance(chan, MemorySlot)][0]
                if mem_slot_index != qubit_mem_slots[qubit]:
                    raise KeyError('The measurement calibration is not defined on the requested classical bits')
                sched |= meas_q
                del qubit_mem_slots[qubit]
                acquire_excludes[qubit] = mem_slot_index
            except KeyError:
                pass
    if qubit_mem_slots:
        qubits = list(qubit_mem_slots.keys())
        qubit_mem_slots.update(acquire_excludes)
        meas_sched = measure(qubits=qubits, backend=backend, inst_map=inst_map, meas_map=schedule_config.meas_map, qubit_mem_slots=qubit_mem_slots)
        meas_sched = target_qobj_transform(meas_sched)
        meas_sched = meas_sched.exclude(channels=[AcquireChannel(qubit) for qubit in acquire_excludes])
        sched |= meas_sched
    qubit_mem_slots.clear()
    return CircuitPulseDef(schedule=sched, qubits=[chan.index for chan in sched.channels if isinstance(chan, AcquireChannel)])
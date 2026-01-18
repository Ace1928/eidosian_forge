from typing import List, Optional, Sequence
import cirq
def estimate_run_sweep_time(program: cirq.AbstractCircuit, params: cirq.Sweepable=None, repetitions: int=1000, latency: Optional[float]=_BASE_LATENCY) -> float:
    """Compute the estimated time for running a parameter sweep across a single Circuit.

    This should approximate, in seconds, the time for the execution of a batch of circuits
    using Engine.run_sweep() on QCS at a time where there is no queue (such as a reserved slot).
    This estimation should be considered a rough approximation.  Many factors can contribute to
    the execution time of a circuit, and the run time can also vary as the service's code changes
    frequently.

    Args:
        program: circuit to be executed
        params: a parameter sweep of variable resolvers to use with the circuit
        repetitions: number of repetitions to execute per parameter sweep
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    width = len(program.all_qubits())
    depth = len(program)
    sweeps = len(list(cirq.to_resolvers(params)))
    return _estimate_run_time_seconds(width, depth, sweeps, repetitions, latency)
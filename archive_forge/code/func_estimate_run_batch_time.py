from typing import List, Optional, Sequence
import cirq
def estimate_run_batch_time(programs: Sequence[cirq.AbstractCircuit], params_list: List[cirq.Sweepable], repetitions: int=1000, latency: float=_BASE_LATENCY) -> float:
    """Compute the estimated time for running a batch of programs.

    This should approximate, in seconds, the time for the execution of a batch of circuits
    using Engine.run_batch() on QCS at a time where there is no queue (such as a reserved slot).
    This estimation should be considered a rough approximation.  Many factors can contribute to
    the execution time of a circuit, and the run time can also vary as the service's code changes
    frequently.

    Args:
        programs: a sequence of circuits to be executed
        params_list: a parameter sweep for each circuit
        repetitions: number of repetitions to execute per parameter sweep
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    total_time = 0.0
    current_width = None
    total_depth = 0
    total_sweeps = 0
    num_circuits = 0
    for idx, program in enumerate(programs):
        width = len(program.all_qubits())
        if width != current_width:
            if num_circuits > 0:
                total_time += _estimate_run_time_seconds(width, total_depth // num_circuits, total_sweeps, repetitions, 0.25)
            num_circuits = 0
            total_depth = 0
            total_sweeps = 0
            current_width = width
        total_depth += len(program)
        num_circuits += 1
        total_sweeps += len(list(cirq.to_resolvers(params_list[idx])))
    if num_circuits > 0:
        total_time += _estimate_run_time_seconds(width, total_depth // num_circuits, total_sweeps, repetitions, 0.0)
    return total_time + latency
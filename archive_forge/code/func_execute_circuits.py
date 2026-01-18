from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def execute_circuits(*, device_graph: nx.Graph, samplers: List[cirq.Sampler], circuits: List[Tuple[cirq.Circuit, List[int]]], routing_attempts: int, compiler: Optional[Callable[[cirq.Circuit], cirq.Circuit]]=None, repetitions: int=10000, add_readout_error_correction=False) -> List[QuantumVolumeResult]:
    """Executes the given circuits on the given samplers.

    Args
        device_graph: The device graph to run the compiled circuit on.
        samplers: The samplers to run the algorithm on.
        circuits: The circuits to sample from.
        routing_attempts: See doc for calculate_quantum_volume.
        compiler: An optional function to compiler the model circuit's
            gates down to the target devices gate set and the optimize it.
        repetitions: The number of bitstrings to sample per circuit.
        add_readout_error_correction: If true, add some parity bits that will
            later be used to detect readout error.

    Returns:
        A list of QuantumVolumeResults that contains all of the information for
        running the algorithm and its results.

    """
    print('Compiling model circuits')
    compiled_circuits: List[CompilationResult] = []
    for idx, (model_circuit, heavy_set) in enumerate(circuits):
        print(f'  Compiling model circuit #{idx + 1}')
        compiled_circuits.append(compile_circuit(model_circuit, device_graph=device_graph, compiler=compiler, routing_attempts=routing_attempts, add_readout_error_correction=add_readout_error_correction))
    results = []
    print('Running samplers over compiled circuits')
    for sampler_i, sampler in enumerate(samplers):
        print(f'  Running sampler #{sampler_i + 1}')
        for circuit_i, compilation_result in enumerate(compiled_circuits):
            model_circuit, heavy_set = circuits[circuit_i]
            prob = sample_heavy_set(compilation_result, heavy_set, repetitions=repetitions, sampler=sampler)
            print(f'    Compiled HOG probability #{circuit_i + 1}: {prob}')
            results.append(QuantumVolumeResult(model_circuit=model_circuit, heavy_set=heavy_set, compiled_circuit=compilation_result.circuit, sampler_result=prob))
    return results
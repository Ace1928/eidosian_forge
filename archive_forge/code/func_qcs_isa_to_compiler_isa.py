from qcs_api_client.models import InstructionSetArchitecture, Characteristic, Operation
from pyquil.external.rpcq import CompilerISA, add_edge, add_qubit, get_qubit, get_edge
import numpy as np
from pyquil.external.rpcq import (
from typing import List, Union, cast, DefaultDict, Set, Optional
from collections import defaultdict
def qcs_isa_to_compiler_isa(isa: InstructionSetArchitecture) -> CompilerISA:
    device = CompilerISA()
    for node in isa.architecture.nodes:
        add_qubit(device, node.node_id)
    for edge in isa.architecture.edges:
        add_edge(device, edge.node_ids[0], edge.node_ids[1])
    qubit_operations_seen: DefaultDict[int, Set[str]] = defaultdict(set)
    edge_operations_seen: DefaultDict[str, Set[str]] = defaultdict(set)
    for operation in isa.instructions:
        for site in operation.sites:
            if operation.node_count == 1:
                if len(site.node_ids) != 1:
                    raise QCSISAParseError(f'operation {operation.name} has node count 1, but site has {len(site.node_ids)} node_ids')
                operation_qubit = get_qubit(device, site.node_ids[0])
                if operation_qubit is None:
                    raise QCSISAParseError(f'operation {operation.name} has node {site.node_ids[0]} but node not declared in architecture')
                if operation.name in qubit_operations_seen[operation_qubit.id]:
                    continue
                qubit_operations_seen[operation_qubit.id].add(operation.name)
                operation_qubit.gates.extend(_transform_qubit_operation_to_gates(operation.name, operation_qubit.id, site.characteristics, isa.benchmarks))
            elif operation.node_count == 2:
                if len(site.node_ids) != 2:
                    QCSISAParseError(f'operation {operation.name} has node count 2, but site has {len(site.node_ids)} node_ids')
                operation_edge = get_edge(device, site.node_ids[0], site.node_ids[1])
                edge_id = make_edge_id(site.node_ids[0], site.node_ids[1])
                if operation_edge is None:
                    raise QCSISAParseError(f'operation {operation.name} has site {site.node_ids}, but edge {edge_id} not declared in architecture')
                if operation.name in edge_operations_seen[edge_id]:
                    continue
                edge_operations_seen[edge_id].add(operation.name)
                operation_edge.gates.extend(_transform_edge_operation_to_gates(operation.name, site.characteristics))
            else:
                raise QCSISAParseError('unexpected operation node count: {}'.format(operation.node_count))
    for qubit in device.qubits.values():
        if len(qubit.gates) == 0:
            qubit.dead = True
    for edge in device.edges.values():
        if len(edge.gates) == 0:
            edge.dead = True
    return device
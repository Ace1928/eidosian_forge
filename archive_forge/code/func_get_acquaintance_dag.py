from typing import FrozenSet, Sequence, Set, TYPE_CHECKING
from cirq import devices
from cirq.contrib.acquaintance.executor import AcquaintanceOperation, ExecutionStrategy
from cirq.contrib.acquaintance.mutation_utils import expose_acquaintance_gates
from cirq.contrib.acquaintance.permutation import LogicalIndex, LogicalMapping
from cirq.contrib import circuitdag
def get_acquaintance_dag(strategy: 'cirq.Circuit', initial_mapping: LogicalMapping):
    strategy = strategy.copy()
    expose_acquaintance_gates(strategy)
    LogicalAnnotator(initial_mapping)(strategy)
    acquaintance_ops = (op for moment in strategy._moments for op in moment.operations if isinstance(op, AcquaintanceOperation))
    return circuitdag.CircuitDag.from_ops(acquaintance_ops)
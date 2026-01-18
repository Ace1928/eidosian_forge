import collections
from typing import Optional
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import Collect1qRuns
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import GateDirection
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit.transpiler.passes import ASAPScheduleAnalysis
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import FilterOpNodes
from qiskit.transpiler.passes import ValidatePulseGates
from qiskit.transpiler.passes import PadDelay
from qiskit.transpiler.passes import InstructionDurationCheck
from qiskit.transpiler.passes import ConstrainedReschedule
from qiskit.transpiler.passes import PulseGates
from qiskit.transpiler.passes import ContainsInstruction
from qiskit.transpiler.passes import VF2PostLayout
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
def generate_translation_passmanager(target, basis_gates=None, method='translator', approximation_degree=None, coupling_map=None, backend_props=None, unitary_synthesis_method='default', unitary_synthesis_plugin_config=None, hls_config=None):
    """Generate a basis translation :class:`~qiskit.transpiler.PassManager`

    Args:
        target (Target): the :class:`~.Target` object representing the backend
        basis_gates (list): A list of str gate names that represent the basis
            gates on the backend target
        method (str): The basis translation method to use
        approximation_degree (Optional[float]): The heuristic approximation degree to
            use. Can be between 0 and 1.
        coupling_map (CouplingMap): the coupling map of the backend
            in case synthesis is done on a physical circuit. The
            directionality of the coupling_map will be taken into
            account if pulse_optimize is True/None and natural_direction
            is True/None.
        unitary_synthesis_plugin_config (dict): The optional dictionary plugin
            configuration, this is plugin specific refer to the specified plugin's
            documentation for how to use.
        backend_props (BackendProperties): Properties of a backend to
            synthesize for (e.g. gate fidelities).
        unitary_synthesis_method (str): The unitary synthesis method to use. You can
            see a list of installed plugins with :func:`.unitary_synthesis_plugin_names`.
        hls_config (HLSConfig): An optional configuration class to use for
            :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
            Specifies how to synthesize various high-level objects.

    Returns:
        PassManager: The basis translation pass manager

    Raises:
        TranspilerError: If the ``method`` kwarg is not a valid value
    """
    if method == 'translator':
        unroll = [UnitarySynthesis(basis_gates, approximation_degree=approximation_degree, coupling_map=coupling_map, backend_props=backend_props, plugin_config=unitary_synthesis_plugin_config, method=unitary_synthesis_method, target=target), HighLevelSynthesis(hls_config=hls_config, coupling_map=coupling_map, target=target, use_qubit_indices=True, equivalence_library=sel, basis_gates=basis_gates), BasisTranslator(sel, basis_gates, target)]
    elif method == 'synthesis':
        unroll = [UnitarySynthesis(basis_gates, approximation_degree=approximation_degree, coupling_map=coupling_map, backend_props=backend_props, plugin_config=unitary_synthesis_plugin_config, method=unitary_synthesis_method, min_qubits=3, target=target), HighLevelSynthesis(hls_config=hls_config, coupling_map=coupling_map, target=target, use_qubit_indices=True, basis_gates=basis_gates, min_qubits=3), Unroll3qOrMore(target=target, basis_gates=basis_gates), Collect2qBlocks(), Collect1qRuns(), ConsolidateBlocks(basis_gates=basis_gates, target=target, approximation_degree=approximation_degree), UnitarySynthesis(basis_gates=basis_gates, approximation_degree=approximation_degree, coupling_map=coupling_map, backend_props=backend_props, plugin_config=unitary_synthesis_plugin_config, method=unitary_synthesis_method, target=target), HighLevelSynthesis(hls_config=hls_config, coupling_map=coupling_map, target=target, use_qubit_indices=True, basis_gates=basis_gates)]
    else:
        raise TranspilerError('Invalid translation method %s.' % method)
    return PassManager(unroll)
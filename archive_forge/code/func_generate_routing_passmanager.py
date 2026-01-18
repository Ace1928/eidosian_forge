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
def generate_routing_passmanager(routing_pass, target, coupling_map=None, vf2_call_limit=None, backend_properties=None, seed_transpiler=None, check_trivial=False, use_barrier_before_measurement=True, vf2_max_trials=None):
    """Generate a routing :class:`~qiskit.transpiler.PassManager`

    Args:
        routing_pass (TransformationPass): The pass which will perform the
            routing
        target (Target): the :class:`~.Target` object representing the backend
        coupling_map (CouplingMap): The coupling map of the backend to route
            for
        vf2_call_limit (int): The internal call limit for the vf2 post layout
            pass. If this is ``None`` or ``0`` the vf2 post layout will not be
            run.
        backend_properties (BackendProperties): Properties of a backend to
            synthesize for (e.g. gate fidelities).
        seed_transpiler (int): Sets random seed for the stochastic parts of
            the transpiler.
        check_trivial (bool): If set to true this will condition running the
            :class:`~.VF2PostLayout` pass after routing on whether a trivial
            layout was tried and was found to not be perfect. This is only
            needed if the constructed pass manager runs :class:`~.TrivialLayout`
            as a first layout attempt and uses it if it's a perfect layout
            (as is the case with preset pass manager level 1).
        use_barrier_before_measurement (bool): If true (the default) the
            :class:`~.BarrierBeforeFinalMeasurements` transpiler pass will be run prior to the
            specified pass in the ``routing_pass`` argument.
        vf2_max_trials (int): The maximum number of trials to run VF2 when
            evaluating the vf2 post layout
            pass. If this is ``None`` or ``0`` the vf2 post layout will not be run.
    Returns:
        PassManager: The routing pass manager
    """

    def _run_post_layout_condition(property_set):
        if not check_trivial or _layout_not_perfect(property_set):
            vf2_stop_reason = property_set['VF2Layout_stop_reason']
            if vf2_stop_reason is None or vf2_stop_reason != VF2LayoutStopReason.SOLUTION_FOUND:
                return True
        return False
    routing = PassManager()
    if target is not None:
        routing.append(CheckMap(target, property_set_field='routing_not_needed'))
    else:
        routing.append(CheckMap(coupling_map, property_set_field='routing_not_needed'))

    def _swap_condition(property_set):
        return not property_set['routing_not_needed']
    if use_barrier_before_measurement:
        routing.append(ConditionalController([BarrierBeforeFinalMeasurements(label='qiskit.transpiler.internal.routing.protection.barrier'), routing_pass], condition=_swap_condition))
    else:
        routing.append(ConditionalController(routing_pass, condition=_swap_condition))
    is_vf2_fully_bounded = vf2_call_limit and vf2_max_trials
    if (target is not None or backend_properties is not None) and is_vf2_fully_bounded:
        routing.append(ConditionalController(VF2PostLayout(target, coupling_map, backend_properties, seed_transpiler, call_limit=vf2_call_limit, max_trials=vf2_max_trials, strict_direction=False), condition=_run_post_layout_condition))
        routing.append(ConditionalController(ApplyLayout(), condition=_apply_post_layout_condition))

    def filter_fn(node):
        return getattr(node.op, 'label', None) != 'qiskit.transpiler.internal.routing.protection.barrier'
    routing.append([FilterOpNodes(filter_fn)])
    return routing
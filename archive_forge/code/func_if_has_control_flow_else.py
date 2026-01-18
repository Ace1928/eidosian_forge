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
def if_has_control_flow_else(if_present, if_absent):
    """Generate a pass manager that will run the passes in ``if_present`` if the given circuit
    has control-flow operations in it, and those in ``if_absent`` if it doesn't."""
    if isinstance(if_present, PassManager):
        if_present = if_present.to_flow_controller()
    if isinstance(if_absent, PassManager):
        if_absent = if_absent.to_flow_controller()
    out = PassManager()
    out.append(ContainsInstruction(CONTROL_FLOW_OP_NAMES, recurse=False))
    out.append(ConditionalController(if_present, condition=_has_control_flow))
    out.append(ConditionalController(if_absent, condition=_without_control_flow))
    return out
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import SabreSwap
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import VF2Layout
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.passes import DenseLayout
from qiskit.transpiler.passes import TrivialLayout
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import OptimizeSwapBeforeMeasure
from qiskit.transpiler.passes import RemoveDiagonalGatesBeforeMeasure
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler.preset_passmanagers.plugin import (
from qiskit.transpiler.passes.optimization import (
from qiskit.transpiler.passes import Depth, Size, FixedPoint, MinimumPoint
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis
from qiskit.passmanager.flow_controllers import ConditionalController, DoWhileController
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.circuit.library.standard_gates import (
def pass_manager(self, pass_manager_config, optimization_level=None) -> PassManager:
    _given_layout = SetLayout(pass_manager_config.initial_layout)

    def _choose_layout_condition(property_set):
        return not property_set['layout']

    def _swap_mapped(property_set):
        return property_set['final_layout'] is None
    if pass_manager_config.target is None:
        coupling_map = pass_manager_config.coupling_map
    else:
        coupling_map = pass_manager_config.target
    layout = PassManager()
    layout.append(_given_layout)
    if optimization_level == 0:
        layout_pass = SabreLayout(coupling_map, max_iterations=1, seed=pass_manager_config.seed_transpiler, swap_trials=5, layout_trials=5, skip_routing=pass_manager_config.routing_method is not None and pass_manager_config.routing_method != 'sabre')
    elif optimization_level == 1:
        layout_pass = SabreLayout(coupling_map, max_iterations=2, seed=pass_manager_config.seed_transpiler, swap_trials=5, layout_trials=5, skip_routing=pass_manager_config.routing_method is not None and pass_manager_config.routing_method != 'sabre')
    elif optimization_level == 2:
        layout_pass = SabreLayout(coupling_map, max_iterations=2, seed=pass_manager_config.seed_transpiler, swap_trials=10, layout_trials=10, skip_routing=pass_manager_config.routing_method is not None and pass_manager_config.routing_method != 'sabre')
    elif optimization_level == 3:
        layout_pass = SabreLayout(coupling_map, max_iterations=4, seed=pass_manager_config.seed_transpiler, swap_trials=20, layout_trials=20, skip_routing=pass_manager_config.routing_method is not None and pass_manager_config.routing_method != 'sabre')
    else:
        raise TranspilerError(f'Invalid optimization level: {optimization_level}')
    layout.append(ConditionalController([BarrierBeforeFinalMeasurements('qiskit.transpiler.internal.routing.protection.barrier'), layout_pass], condition=_choose_layout_condition))
    embed = common.generate_embed_passmanager(coupling_map)
    layout.append(ConditionalController(embed.to_flow_controller(), condition=_swap_mapped))
    return layout
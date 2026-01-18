from __future__ import annotations
import os
import inspect
import tempfile
from qiskit.utils import optionals as _optionals
from qiskit.passmanager.base_tasks import BaseController, GenericPass
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError
@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PYDOT.require_in_call
def pass_manager_drawer(pass_manager, filename=None, style=None, raw=False):
    """
    Draws the pass manager.

    This function needs `pydot <https://github.com/pydot/pydot>`__, which in turn needs
    `Graphviz <https://www.graphviz.org/>`__ to be installed.

    Args:
        pass_manager (PassManager): the pass manager to be drawn
        filename (str): file path to save image to
        style (dict or OrderedDict): keys are the pass classes and the values are
            the colors to make them. An example can be seen in the DEFAULT_STYLE. An ordered
            dict can be used to ensure a priority coloring when pass falls into multiple
            categories. Any values not included in the provided dict will be filled in from
            the default dict
        raw (Bool) : True if you want to save the raw Dot output not an image. The
            default is False.
    Returns:
        PIL.Image or None: an in-memory representation of the pass manager. Or None if
        no image was generated or PIL is not installed.
    Raises:
        MissingOptionalLibraryError: when nxpd or pydot not installed.
        VisualizationError: If raw=True and filename=None.

    Example:
        .. code-block::

             %matplotlib inline
            from qiskit import QuantumCircuit
            from qiskit.compiler import transpile
            from qiskit.transpiler import PassManager
            from qiskit.visualization import pass_manager_drawer
            from qiskit.transpiler.passes import Unroller

            circ = QuantumCircuit(3)
            circ.ccx(0, 1, 2)
            circ.draw()

            pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
            pm = PassManager(pass_)
            new_circ = pm.run(circ)
            new_circ.draw(output='mpl')

            pass_manager_drawer(pm, "passmanager.jpg")
    """
    import pydot
    if not style:
        style = DEFAULT_STYLE
    graph = pydot.Dot()
    component_id = 0
    prev_node = None
    for index, controller_group in enumerate(pass_manager.to_flow_controller().tasks):
        subgraph, component_id, prev_node = draw_subgraph(controller_group, component_id, style, prev_node, index)
        graph.add_subgraph(subgraph)
    output = make_output(graph, raw, filename)
    return output
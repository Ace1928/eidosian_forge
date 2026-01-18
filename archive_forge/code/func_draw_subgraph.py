from __future__ import annotations
import os
import inspect
import tempfile
from qiskit.utils import optionals as _optionals
from qiskit.passmanager.base_tasks import BaseController, GenericPass
from qiskit.passmanager.flow_controllers import FlowControllerLinear
from qiskit.transpiler.basepasses import AnalysisPass, TransformationPass
from .exceptions import VisualizationError
def draw_subgraph(controller_group, component_id, style, prev_node, idx):
    """Draw subgraph."""
    import pydot
    label = f'[{idx}] '
    if isinstance(controller_group, BaseController) and (not isinstance(controller_group, FlowControllerLinear)):
        label += f'{controller_group.__class__.__name__}'
    subgraph = pydot.Cluster(str(component_id), label=label, fontname='helvetica', labeljust='l')
    component_id += 1
    if isinstance(controller_group, BaseController):
        tasks = getattr(controller_group, 'tasks', [])
    elif isinstance(controller_group, GenericPass):
        tasks = [controller_group]
    elif isinstance(controller_group, (list, tuple)):
        tasks = controller_group
    else:
        return (subgraph, component_id, prev_node)
    flatten_tasks = []
    for task in tasks:
        if isinstance(task, FlowControllerLinear):
            flatten_tasks.extend(task.tasks)
        else:
            flatten_tasks.append(task)
    for task in flatten_tasks:
        if isinstance(task, BaseController):
            node = pydot.Node(str(component_id), label='Nested flow controller', color='k', shape='rectangle', fontname='helvetica')
        else:
            node = pydot.Node(str(component_id), label=str(type(task).__name__), color=_get_node_color(task, style), shape='rectangle', fontname='helvetica')
        subgraph.add_node(node)
        component_id += 1
        arg_spec = inspect.getfullargspec(task.__init__)
        args = arg_spec[0][1:]
        num_optional = len(arg_spec[3]) if arg_spec[3] else 0
        for arg_index, arg in enumerate(args):
            nd_style = 'solid'
            if arg_index >= len(args) - num_optional:
                nd_style = 'dashed'
            input_node = pydot.Node(component_id, label=arg, color='black', shape='ellipse', fontsize=10, style=nd_style, fontname='helvetica')
            subgraph.add_node(input_node)
            component_id += 1
            subgraph.add_edge(pydot.Edge(input_node, node))
        if prev_node:
            subgraph.add_edge(pydot.Edge(prev_node, node))
        prev_node = node
    return (subgraph, component_id, prev_node)
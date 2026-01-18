import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def find_nested_workflows(workflow):
    """Given a workflow, find nested workflow objects."""
    nested_workflows = []
    wf_nodes = nx.nodes(workflow._graph)
    for node in wf_nodes:
        if isinstance(node, Workflow):
            nested_workflows.append(node)
    return nested_workflows
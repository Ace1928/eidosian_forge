import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def sink_outputs(self, dir_name=None):
    """Connect the outputs of a workflow to a datasink."""
    outputs = self.out_node.outputs.get()
    prefix = '@' if dir_name is None else dir_name + '.@'
    for field in outputs:
        self.wf.connect(self.out_node, field, self.sink_node, prefix + field)
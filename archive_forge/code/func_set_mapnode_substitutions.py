import os.path as op
import networkx as nx
from nipype import Workflow, MapNode, Node, IdentityInterface
from nipype.interfaces.base import (  # BaseInterfaceInputSpec,
def set_mapnode_substitutions(self, n_runs, template_pattern='run_%d', template_args='r + 1'):
    """Find mapnode names and add datasink substitutions to sort by run."""
    mapnode_names = find_mapnodes(self.wf)
    nested_workflows = find_nested_workflows(self.wf)
    for wf in nested_workflows:
        mapnode_names.extend(find_mapnodes(wf))
    substitutions = []
    for r in reversed(range(n_runs)):
        templ_args = eval('(%s)' % template_args)
        for name in mapnode_names:
            substitutions.append(('_%s%d' % (name, r), template_pattern % templ_args))
    if isdefined(self.sink_node.inputs.substitutions):
        self.sink_node.inputs.substitutions.extend(substitutions)
    else:
        self.sink_node.inputs.substitutions = substitutions
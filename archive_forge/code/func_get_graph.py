import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def get_graph(self, title=None, roi_state=None):
    title = title if title else self.machine.title
    fsm_graph = pgv.Digraph(name=title, node_attr=self.machine.style_attributes['node']['default'], edge_attr=self.machine.style_attributes['edge']['default'], graph_attr=self.machine.style_attributes['graph']['default'])
    fsm_graph.graph_attr.update(**self.machine.machine_attributes)
    fsm_graph.graph_attr['label'] = title
    states, transitions = self._get_elements()
    if roi_state:
        transitions = [t for t in transitions if t['source'] == roi_state or self.custom_styles['edge'][t['source']][t['dest']]]
        state_names = [t for trans in transitions for t in [trans['source'], trans.get('dest', trans['source'])]]
        state_names += [k for k, style in self.custom_styles['node'].items() if style]
        states = _filter_states(states, state_names, self.machine.state_cls)
    self._add_nodes(states, fsm_graph)
    self._add_edges(transitions, fsm_graph)
    setattr(fsm_graph, 'draw', partial(self.draw, fsm_graph))
    return fsm_graph
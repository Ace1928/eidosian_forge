from a triple store that implements the Redland storage interface; similarly,
import os
from io import StringIO
from Bio import MissingPythonDependencyError
from Bio.Phylo import CDAO
from ._cdao_owl import cdao_namespaces, resolve_uri
def get_node_info(self, graph, context=None):
    """Create a dictionary containing information about all nodes in the tree."""
    self.node_info = {}
    self.obj_info = {}
    self.children = {}
    self.nodes = set()
    self.tree_roots = set()
    assignments = {qUri('cdao:has_Parent'): 'parent', qUri('cdao:belongs_to_Edge_as_Child'): 'edge', qUri('cdao:has_Annotation'): 'annotation', qUri('cdao:has_Value'): 'value', qUri('cdao:represents_TU'): 'tu', qUri('rdfs:label'): 'label', qUri('cdao:has_Support_Value'): 'confidence'}
    for s, v, o in graph:
        s, v, o = (str(s), str(v), str(o))
        if s not in self.obj_info:
            self.obj_info[s] = {}
        this = self.obj_info[s]
        try:
            this[assignments[v]] = o
        except KeyError:
            pass
        if v == qUri('rdf:type'):
            if o in (qUri('cdao:AncestralNode'), qUri('cdao:TerminalNode')):
                self.nodes.add(s)
        if v == qUri('cdao:has_Root'):
            self.tree_roots.add(o)
    for node in self.nodes:
        self.node_info[node] = {}
        node_info = self.node_info[node]
        obj = self.obj_info[node]
        if 'edge' in obj:
            edge = self.obj_info[obj['edge']]
            if 'annotation' in edge:
                annotation = self.obj_info[edge['annotation']]
                if 'value' in annotation:
                    node_info['branch_length'] = float(annotation['value'])
        if 'tu' in obj:
            tu = self.obj_info[obj['tu']]
            if 'label' in tu:
                node_info['label'] = tu['label']
        if 'parent' in obj:
            parent = obj['parent']
            if parent not in self.children:
                self.children[parent] = []
            self.children[parent].append(node)
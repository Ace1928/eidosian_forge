import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def add_slices(self, data, node_or_edge_xml):
    slices_element = node_or_edge_xml.find(f'{{{self.NS_GEXF}}}slices')
    if slices_element is not None:
        data['slices'] = []
        for s in slices_element.findall(f'{{{self.NS_GEXF}}}slice'):
            start = s.get('start')
            end = s.get('end')
            data['slices'].append((start, end))
    return data
import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def add_spells(self, data, node_or_edge_xml):
    spells_element = node_or_edge_xml.find(f'{{{self.NS_GEXF}}}spells')
    if spells_element is not None:
        data['spells'] = []
        ttype = self.timeformat
        for s in spells_element.findall(f'{{{self.NS_GEXF}}}spell'):
            start = self.python_type[ttype](s.get('start'))
            end = self.python_type[ttype](s.get('end'))
            data['spells'].append((start, end))
    return data
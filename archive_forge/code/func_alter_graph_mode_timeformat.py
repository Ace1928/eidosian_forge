import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def alter_graph_mode_timeformat(self, start_or_end):
    if self.graph_element.get('mode') == 'static':
        if start_or_end is not None:
            if isinstance(start_or_end, str):
                timeformat = 'date'
            elif isinstance(start_or_end, float):
                timeformat = 'double'
            elif isinstance(start_or_end, int):
                timeformat = 'long'
            else:
                raise nx.NetworkXError('timeformat should be of the type int, float or str')
            self.graph_element.set('timeformat', timeformat)
            self.graph_element.set('mode', 'dynamic')
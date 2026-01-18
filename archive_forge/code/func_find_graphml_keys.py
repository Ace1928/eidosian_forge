import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def find_graphml_keys(self, graph_element):
    """Extracts all the keys and key defaults from the xml."""
    graphml_keys = {}
    graphml_key_defaults = {}
    for k in graph_element.findall(f'{{{self.NS_GRAPHML}}}key'):
        attr_id = k.get('id')
        attr_type = k.get('attr.type')
        attr_name = k.get('attr.name')
        yfiles_type = k.get('yfiles.type')
        if yfiles_type is not None:
            attr_name = yfiles_type
            attr_type = 'yfiles'
        if attr_type is None:
            attr_type = 'string'
            warnings.warn(f'No key type for id {attr_id}. Using string')
        if attr_name is None:
            raise nx.NetworkXError(f'Unknown key for id {attr_id}.')
        graphml_keys[attr_id] = {'name': attr_name, 'type': self.python_type[attr_type], 'for': k.get('for')}
        default = k.find(f'{{{self.NS_GRAPHML}}}default')
        if default is not None:
            python_type = graphml_keys[attr_id]['type']
            if python_type == bool:
                graphml_key_defaults[attr_id] = self.convert_bool[default.text.lower()]
            else:
                graphml_key_defaults[attr_id] = python_type(default.text)
    return (graphml_keys, graphml_key_defaults)
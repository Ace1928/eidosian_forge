import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def get_key(self, name, attr_type, scope, default):
    keys_key = (name, attr_type, scope)
    try:
        return self.keys[keys_key]
    except KeyError:
        if self.named_key_ids:
            new_id = name
        else:
            new_id = f'd{len(list(self.keys))}'
        self.keys[keys_key] = new_id
        key_kwargs = {'id': new_id, 'for': scope, 'attr.name': name, 'attr.type': attr_type}
        key_element = self.myElement('key', **key_kwargs)
        if default is not None:
            default_element = self.myElement('default')
            default_element.text = str(default)
            key_element.append(default_element)
        self.xml.insert(0, key_element)
    return new_id
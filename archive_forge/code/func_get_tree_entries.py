from jedi.inference import compiled
from jedi.inference import analysis
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues, \
from jedi.inference.helpers import get_int_or_none, is_string, \
from jedi.inference.utils import safe_property, to_list
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.filters import LazyAttributeOverwrite, publish_method
from jedi.inference.base_value import ValueSet, Value, NO_VALUES, \
from jedi.parser_utils import get_sync_comp_fors
from jedi.inference.context import CompForContext
from jedi.inference.value.dynamic_arrays import check_array_additions
def get_tree_entries(self):
    c = self.atom.children
    if self.atom.type in self._TUPLE_LIKE:
        return c[::2]
    array_node = c[1]
    if array_node in (']', '}', ')'):
        return []
    if array_node.type == 'testlist_comp':
        return [value for value in array_node.children[::2] if value.type != 'star_expr']
    elif array_node.type == 'dictorsetmaker':
        kv = []
        iterator = iter(array_node.children)
        for key in iterator:
            if key == '**':
                next(iterator)
                next(iterator, None)
            else:
                op = next(iterator, None)
                if op is None or op == ',':
                    if key.type == 'star_expr':
                        pass
                    else:
                        kv.append(key)
                else:
                    assert op == ':'
                    kv.append((key, next(iterator)))
                    next(iterator, None)
        return kv
    elif array_node.type == 'star_expr':
        return []
    else:
        return [array_node]
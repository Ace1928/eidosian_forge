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
def comprehension_from_atom(inference_state, value, atom):
    bracket = atom.children[0]
    test_list_comp = atom.children[1]
    if bracket == '{':
        if atom.children[1].children[1] == ':':
            sync_comp_for = test_list_comp.children[3]
            if sync_comp_for.type == 'comp_for':
                sync_comp_for = sync_comp_for.children[1]
            return DictComprehension(inference_state, value, sync_comp_for_node=sync_comp_for, key_node=test_list_comp.children[0], value_node=test_list_comp.children[2])
        else:
            cls = SetComprehension
    elif bracket == '(':
        cls = GeneratorComprehension
    elif bracket == '[':
        cls = ListComprehension
    sync_comp_for = test_list_comp.children[1]
    if sync_comp_for.type == 'comp_for':
        sync_comp_for = sync_comp_for.children[1]
    return cls(inference_state, defining_context=value, sync_comp_for_node=sync_comp_for, entry_node=test_list_comp.children[0])
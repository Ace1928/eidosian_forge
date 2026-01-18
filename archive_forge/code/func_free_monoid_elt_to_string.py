import string
from ..sage_helper import _within_sage, sage_method
def free_monoid_elt_to_string(elt):
    vars = elt.parent().variable_names()
    return ''.join([e * vars[v] for v, e in elt._element_list])
import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
def num_self_crossings(component):
    comp_set = set(component)
    return len([1 for ce in component if ce.other() in comp_set])
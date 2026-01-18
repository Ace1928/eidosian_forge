import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
def simplify_func(link):
    if isinstance(simplify, dict):
        return link.simplify(**simplify)
    elif isinstance(simplify, str):
        return link.simplify(mode=simplify)
    return False
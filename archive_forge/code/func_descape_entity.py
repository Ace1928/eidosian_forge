import html
import re
from collections import defaultdict
def descape_entity(m, defs=html.entities.entitydefs):
    """
    Translate one entity to its ISO Latin value.
    Inspired by example from effbot.org


    """
    try:
        return defs[m.group(1)]
    except KeyError:
        return m.group(0)
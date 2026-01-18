import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def fill_in_crossings(link, sides):
    """
    Given boundary as above , fill in crossings on either side from sides.
    Returns a dictionary with the side (0 or 1) of each crossing.
    """
    crossing_sides = dict([(x[0], sides[x]) for x in sides])
    crossing_labels = map(lambda c: c.label, link.crossings)
    crossings_to_sort = set(crossing_labels) - set((x[0] for x in sides))
    while crossings_to_sort:
        start_crossing = crossings_to_sort.pop()
        accumulated_crossings = [start_crossing]
        m, end_side = meander(crossing_strand_from_name(link, (start_crossing, randint(0, 3))), sides)
        accumulated_crossings.extend(map(lambda x: x.label, m))
        for c in accumulated_crossings:
            crossing_sides[c] = end_side
    return crossing_sides
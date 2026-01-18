import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def meander(cs, sides):
    """
    Wander randomly starting at crossing strand cs until you hit
    a boundary strand in sides.Assumes the crossing of cs is not on the side.
    Returns a set of all the crossings encountered along the way,
    including ones on the boundary, and which side (0 or 1) is hit
    """
    crossings_encountered = [cs.crossing]
    end_side = 0
    while True:
        cs = cs.opposite().rotate(randint(1, 3))
        if cslabel(cs) in sides:
            end_side = sides[cslabel(cs)]
            break
        crossings_encountered.append(cs.crossing)
    return (set(crossings_encountered), end_side)
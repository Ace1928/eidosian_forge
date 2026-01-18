import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def crossing_ball(crossing, radius):
    """
    Returns the crossings within distance r of the crossing, in the form
    of a dictionary, where the values are the distances to the center crossing,
    and a list of the crossing strands along the boundary.
    """
    distances = {crossing: 0}
    opposite_positions = [cs.opposite() for cs in crossing.crossing_strands() if cs.opposite().crossing != crossing]
    for i in range(1, radius):
        new_opposites = []
        for cs in opposite_positions:
            if cs.crossing not in distances:
                distances[cs.crossing] = i
            new_opposites.append(cs.rotate(1).opposite())
            new_opposites.append(cs.rotate(2).opposite())
            new_opposites.append(cs.rotate(3).opposite())
        op_repeats = [x for x in new_opposites if x.crossing not in distances]
        opposite_positions = []
        for cs in op_repeats:
            if cs not in opposite_positions:
                opposite_positions.append(cs)
    return (distances, map(lambda x: x.opposite(), opposite_positions))
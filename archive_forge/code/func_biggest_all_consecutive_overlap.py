from .links_base import Strand, Crossing, Link
import random
import collections
def biggest_all_consecutive_overlap(self):
    """
        Return a random crossing from among those with the maximal possible
        overlap.
        """
    overlap_indices = collections.defaultdict(list)
    for i, cs in self.int_to_set.items():
        overlap_indices[cs.opposite()[0]].append(i)
    possible = []
    for crossing, overlap in overlap_indices.items():
        overlap = sorted(overlap)
        if is_range(overlap):
            possible.append((len(overlap), min(overlap), crossing))
    max_overlap = max(possible)[0]
    good_choices = [pos for pos in possible if pos[0] == max_overlap]
    return random.choice(good_choices)
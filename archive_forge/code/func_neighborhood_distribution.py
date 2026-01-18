import spherogram
from spherogram.links.tangles import Tangle, OneTangle, MinusOneTangle
import networkx as nx
from random import randint,choice,sample
from spherogram.links.random_links import map_to_link, random_map
def neighborhood_distribution(link, radius):
    nhds = all_neighborhoods(link, radius)
    nhd_classes = []
    for nhd in nhds:
        already_found = False
        for nhd_class in nhd_classes:
            if nhd.is_isotopic(nhd_class[0]):
                nhd_class[1] += 1
                already_found = True
                break
        if not already_found:
            nhd_classes.append([nhd, 1])
    return nhd_classes
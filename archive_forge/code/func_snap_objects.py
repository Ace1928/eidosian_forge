import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def snap_objects(objs, attr: str, tolerance) -> list:
    axis = {'x0': 'h', 'x1': 'h', 'top': 'v', 'bottom': 'v'}[attr]
    list_objs = list(objs)
    clusters = cluster_objects(list_objs, itemgetter(attr), tolerance)
    avgs = [sum(map(itemgetter(attr), cluster)) / len(cluster) for cluster in clusters]
    snapped_clusters = [[move_object(obj, axis, avg - obj[attr]) for obj in cluster] for cluster, avg in zip(clusters, avgs)]
    return list(itertools.chain(*snapped_clusters))
import collections
from absl import logging
def graph_rewrites():
    return collections.namedtuple('GraphRewrites', ['enabled', 'disabled', 'default'])
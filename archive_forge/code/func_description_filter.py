import itertools
from ._init_environment import SetHostMTurkConnection
from ._init_environment import config_environment
def description_filter(substring):
    return lambda hit: substring in hit.Title
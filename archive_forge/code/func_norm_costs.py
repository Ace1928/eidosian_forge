from . import schema
from .jsonutil import get_column
from .search import Search
def norm_costs(costs, norm=1000):
    max_cost = max(costs)
    return [cost / max_cost * norm for cost in costs]
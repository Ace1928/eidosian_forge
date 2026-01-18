import copy
from oslo_config import cfg
from oslo_log import log as logging
import stevedore
from glance.i18n import _, _LE
from the first responsive active location it finds in this list.
def get_ordered_locations(locations, **kwargs):
    """
    Order image location list by configured strategy.

    :param locations: The original image location list.
    :param kwargs: Strategy-specific arguments for under layer strategy module.
    :returns: The image location list with strategy-specific order.
    """
    if not locations:
        return []
    strategy_module = _available_strategies[CONF.location_strategy]
    return strategy_module.get_ordered_locations(copy.deepcopy(locations), **kwargs)
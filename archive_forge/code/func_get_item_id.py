import json
import os
import numpy as np
def get_item_id(item: str) -> int:
    """
    Gets the item ID of an MC item.
    :param item: The item string
    :return: The internal ID of the item.
    """
    if not item.startswith('minecraft:'):
        item = 'minecraft:' + item
    return MC_ITEM_IDS.index(item)
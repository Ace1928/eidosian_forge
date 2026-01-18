import json
import os
import numpy as np
def sort_recipes_by_output(json):
    result = {item: [] for item in ALL_ITEMS}
    for recipe in json:
        if len(recipe['ingredients']) == 0:
            continue
        if recipe['outputItemName'] in recipe['ingredients']:
            continue
        result[recipe['outputItemName']].append(recipe)
    for item, recipes in result.items():
        result[item] = dedup_list(recipes)
    return result
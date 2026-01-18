import itertools
import collections
def splitDict(data):
    """
    Split a dictionary with lists as the data, into smaller dictionaries

    :param data: A dictionary with lists as the values

    :return: A tuple of dictionaries each containing the data separately,
            with the same dictionary keys
    """
    maxitems = max([len(values) for values in data.values()])
    output = [dict() for _ in range(maxitems)]
    for key, values in data.items():
        for i, val in enumerate(values):
            output[i][key] = val
    return tuple(output)
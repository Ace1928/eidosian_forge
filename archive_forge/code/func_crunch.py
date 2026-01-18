import json
from collections.abc import Mapping
def crunch(data):
    index = {}
    values = []
    flatten(data, index, values)
    return values
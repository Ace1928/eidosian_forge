from itertools import chain
import json
import re
import click
def coords_from_query(query):
    """Transform a query line into a (lng, lat) pair of coordinates."""
    try:
        coords = json.loads(query)
    except ValueError:
        query = query.replace(',', ' ')
        vals = query.split()
        coords = [float(v) for v in vals]
    return tuple(coords[:2])
import re
import traitlets
import datetime as dt
def naive_from_json(js, manager):
    """Deserialize a naive Python datetime object from json."""
    if js is None:
        return None
    else:
        return dt.datetime(js['year'], js['month'] + 1, js['date'], js['hours'], js['minutes'], js['seconds'], js['milliseconds'] * 1000)
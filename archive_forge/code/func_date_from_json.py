import re
import traitlets
import datetime as dt
def date_from_json(js, manager):
    """Deserialize a Javascript date."""
    if js is None:
        return None
    else:
        return dt.date(js['year'], js['month'] + 1, js['date'])
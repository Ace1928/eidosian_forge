import datetime
def delta_time(dt, **kwargs):
    """
    Add to the time.
    Ref: https://docs.python.org/3.6/library/datetime.html#timedelta-objects
    """
    return dt + datetime.timedelta(**kwargs)
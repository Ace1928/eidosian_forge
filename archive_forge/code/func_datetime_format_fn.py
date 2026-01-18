import datetime
from typing import Protocol, Tuple, Type, Union
def datetime_format_fn(sequence: str) -> datetime.datetime:
    return datetime.datetime.strptime(sequence, '%Y-%m-%d %H:%M:%S')
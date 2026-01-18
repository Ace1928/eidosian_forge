import datetime
from typing import Protocol, Tuple, Type, Union
def python_types_to_regex(python_type: Type) -> Tuple[str, FormatFunction]:
    if python_type == float:

        def float_format_fn(sequence: str) -> float:
            return float(sequence)
        return (FLOAT, float_format_fn)
    elif python_type == int:

        def int_format_fn(sequence: str) -> int:
            return int(sequence)
        return (INTEGER, int_format_fn)
    elif python_type == bool:

        def bool_format_fn(sequence: str) -> bool:
            return bool(sequence)
        return (BOOLEAN, bool_format_fn)
    elif python_type == datetime.date:

        def date_format_fn(sequence: str) -> datetime.date:
            return datetime.datetime.strptime(sequence, '%Y-%m-%d').date()
        return (DATE, date_format_fn)
    elif python_type == datetime.time:

        def time_format_fn(sequence: str) -> datetime.time:
            return datetime.datetime.strptime(sequence, '%H:%M:%S').time()
        return (TIME, time_format_fn)
    elif python_type == datetime.datetime:

        def datetime_format_fn(sequence: str) -> datetime.datetime:
            return datetime.datetime.strptime(sequence, '%Y-%m-%d %H:%M:%S')
        return (DATETIME, datetime_format_fn)
    else:
        raise NotImplementedError(f'The Python type {python_type} is not supported. Please open an issue.')
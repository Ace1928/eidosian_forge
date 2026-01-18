import pandas
from pandas.api.types import union_categoricals
from modin.error_message import ErrorMessage
def is_part_empty(part):
    return part.empty and (not isinstance(part, pandas.DataFrame) or len(part.columns) == 0)
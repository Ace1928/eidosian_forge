import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def decode_rows(self, stream, conversors):
    for row in stream:
        values = _parse_values(row)
        if not isinstance(values, dict):
            raise BadLayout()
        try:
            yield {key: None if value is None else conversors[key](value) for key, value in values.items()}
        except ValueError as exc:
            if 'float: ' in str(exc):
                raise BadNumericalValue()
            raise
        except IndexError:
            raise BadDataFormat(row)
import re
from typing import Dict
from isoduration.constants import PERIOD_PREFIX, TIME_PREFIX, WEEK_PREFIX
from isoduration.parser.exceptions import OutOfDesignators
def parse_designator(designators: Dict[str, str], target: str) -> str:
    while True:
        try:
            key, value = designators.popitem(last=False)
        except KeyError as exc:
            raise OutOfDesignators from exc
        if key == target:
            return value
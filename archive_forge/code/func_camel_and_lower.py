import json
from pydeck.types.base import PydeckType
def camel_and_lower(w):
    return lower_first_letter(to_camel_case(w))
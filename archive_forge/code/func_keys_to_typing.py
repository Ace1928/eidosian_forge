import socket
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from selenium.types import AnyKey
from selenium.webdriver.common.keys import Keys
def keys_to_typing(value: Iterable[AnyKey]) -> List[str]:
    """Processes the values that will be typed in the element."""
    characters: List[str] = []
    for val in value:
        if isinstance(val, Keys):
            characters.append(val)
        elif isinstance(val, (int, float)):
            characters.extend(str(val))
        else:
            characters.extend(val)
    return characters
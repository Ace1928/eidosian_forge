from typing import Any, Iterable
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.ismock import ismock
def append_list(self, start: str, separator: str, end: str, list: Iterable[Any]) -> Description:
    separate = False
    self.append(start)
    for item in list:
        if separate:
            self.append(separator)
        self.append_description_of(item)
        separate = True
    self.append(end)
    return self
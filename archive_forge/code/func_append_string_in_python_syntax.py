from typing import Any, Iterable
from hamcrest.core.description import Description
from hamcrest.core.helpers.hasmethod import hasmethod
from hamcrest.core.helpers.ismock import ismock
def append_string_in_python_syntax(self, string: str) -> None:
    self.append("'")
    for ch in string:
        self.append(character_in_python_syntax(ch))
    self.append("'")
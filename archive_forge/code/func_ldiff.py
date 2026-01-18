import difflib
import pathlib
from typing import Any, List, Union
def ldiff(self, other: Union[str, pathlib.Path]) -> List[str]:
    return self._diff(self.path, pathlib.Path(other))
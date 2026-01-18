import difflib
import pathlib
from typing import Any, List, Union
def rdiff(self, other: Union[str, pathlib.Path]) -> List[str]:
    return self._diff(pathlib.Path(other), self.path)
import ast
from typing import Optional, Union
def set_source_code(self, src: Optional[str]):
    self.src = src
    self.message = self._format_message()
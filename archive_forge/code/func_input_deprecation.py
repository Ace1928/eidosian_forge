from textwrap import dedent
from typing import Any, Dict, List, Optional, Union
from ..language import DirectiveLocation
def input_deprecation(string: str) -> Optional[str]:
    return string if input_value_deprecation else ''
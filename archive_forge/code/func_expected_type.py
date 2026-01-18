import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
@property
def expected_type(self) -> Union[List[Type[dict]], Tuple[Type[str], Type[str]], Dict[Tuple[Type[str], Type[str]], Type[dict]], List[Tuple[Type[str], Type[str]]]]:
    """
        The type PyGithub expected
        """
    return self.__expectedType
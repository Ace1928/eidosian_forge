import typing
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, List, Literal, Optional, Tuple, Union
from .params import Description, ParamArg, ParamVal
def get_default_key(self) -> Optional[str]:
    """Get the default key for this level of the foldermap.
        Raises a ValueError if it does not have a default.
        """
    return self.__curr_level.get('__default')
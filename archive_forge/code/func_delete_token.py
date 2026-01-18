import warnings
from pathlib import Path
from typing import Optional
from .. import constants
from ._token import get_token
@classmethod
def delete_token(cls) -> None:
    """
        Deletes the token from storage. Does not fail if token does not exist.
        """
    try:
        cls.path_token.unlink()
    except FileNotFoundError:
        pass
    try:
        cls._old_path_token.unlink()
    except FileNotFoundError:
        pass
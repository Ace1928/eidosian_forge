from typing import Dict, Optional, Union
from .. import constants
from ._runtime import (
from ._token import get_token
from ._validators import validate_hf_hub_args
def get_token_to_send(token: Optional[Union[bool, str]]) -> Optional[str]:
    """Select the token to send from either `token` or the cache."""
    if isinstance(token, str):
        return token
    if token is False:
        return None
    cached_token = get_token()
    if token is True:
        if cached_token is None:
            raise LocalTokenNotFoundError('Token is required (`token=True`), but no token found. You need to provide a token or be logged in to Hugging Face with `huggingface-cli login` or `huggingface_hub.login`. See https://huggingface.co/settings/tokens.')
        return cached_token
    if constants.HF_HUB_DISABLE_IMPLICIT_TOKEN:
        return None
    return cached_token
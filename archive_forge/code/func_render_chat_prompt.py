from functools import lru_cache
from typing import Callable, Dict, List, Optional, Union
from ..utils import HfHubHTTPError, RepositoryNotFoundError, is_minijinja_available
def render_chat_prompt(*, model_id: str, messages: List[Dict[str, str]], token: Union[str, bool, None]=None, add_generation_prompt: bool=True, **kwargs) -> str:
    """Render a chat prompt using a model's chat template.

    Args:
        model_id (`str`):
            The model id.
        messages (`List[Dict[str, str]]`):
            The list of messages to render.
        token (`str` or `bool`, *optional*):
            Hugging Face token. Will default to the locally saved token if not provided.

    Returns:
        `str`: The rendered chat prompt.

    Raises:
        `TemplateError`: If there's any issue while fetching, compiling or rendering the chat template.
    """
    minijinja = _import_minijinja()
    template = _fetch_and_compile_template(model_id=model_id, token=token)
    try:
        return template(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)
    except minijinja.TemplateError as e:
        raise TemplateError(f"Error while trying to render chat prompt for model '{model_id}': {e}") from e
import logging
from typing import Any
from uvicorn._types import (
from uvicorn.logging import TRACE_LOG_LEVEL
def message_with_placeholders(message: Any) -> Any:
    """
    Return an ASGI message, with any body-type content omitted and replaced
    with a placeholder.
    """
    new_message = message.copy()
    for attr in PLACEHOLDER_FORMAT.keys():
        if message.get(attr) is not None:
            content = message[attr]
            placeholder = PLACEHOLDER_FORMAT[attr].format(length=len(content))
            new_message[attr] = placeholder
    return new_message
from __future__ import annotations
import typing
@property
def messages_dict(self) -> dict[str, typing.Any]:
    if not isinstance(self.messages, dict):
        raise TypeError("cannot access 'messages_dict' when 'messages' is of type " + type(self.messages).__name__)
    return self.messages
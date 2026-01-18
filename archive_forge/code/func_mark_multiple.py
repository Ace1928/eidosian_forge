from typing import List
from .keymap import KEYMAP, get_character
def mark_multiple(*keys: List[str]):
    """
    Mark the function with the key codes so it can be handled in the register
    """

    def decorator(func):
        handle = getattr(func, 'handle_key', [])
        handle += keys
        func.handle_key = handle
        return func
    return decorator
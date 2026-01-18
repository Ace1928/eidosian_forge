import abc
import re
import typing
def message_fnc(exception: BaseException) -> bool:
    return message == str(exception)
import math
import typing
import uuid
def register_url_convertor(key: str, convertor: Convertor[typing.Any]) -> None:
    CONVERTOR_TYPES[key] = convertor
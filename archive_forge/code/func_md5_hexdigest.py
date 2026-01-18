import hashlib
def md5_hexdigest(data: bytes, *, usedforsecurity: bool=True) -> str:
    return hashlib.md5(data).hexdigest()
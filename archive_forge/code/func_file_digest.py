def file_digest(fileobj, digest, /, *, _bufsize=2 ** 18):
    """Hash the contents of a file-like object. Returns a digest object.

    *fileobj* must be a file-like object opened for reading in binary mode.
    It accepts file objects from open(), io.BytesIO(), and SocketIO objects.
    The function may bypass Python's I/O and use the file descriptor *fileno*
    directly.

    *digest* must either be a hash algorithm name as a *str*, a hash
    constructor, or a callable that returns a hash object.
    """
    if isinstance(digest, str):
        digestobj = new(digest)
    else:
        digestobj = digest()
    if hasattr(fileobj, 'getbuffer'):
        digestobj.update(fileobj.getbuffer())
        return digestobj
    if not (hasattr(fileobj, 'readinto') and hasattr(fileobj, 'readable') and fileobj.readable()):
        raise ValueError(f"'{fileobj!r}' is not a file-like object in binary reading mode.")
    buf = bytearray(_bufsize)
    view = memoryview(buf)
    while True:
        size = fileobj.readinto(buf)
        if size == 0:
            break
        digestobj.update(view[:size])
    return digestobj
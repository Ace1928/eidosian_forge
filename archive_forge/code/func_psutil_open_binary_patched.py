import io
import platform
def psutil_open_binary_patched(fname, *args, **kwargs):
    f = psutil_open_binary(fname, *args, **kwargs)
    if fname == '/proc/meminfo':
        with f:
            return io.BytesIO(f.read().replace(b':', b': '))
    return f
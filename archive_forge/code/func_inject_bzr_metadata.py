from io import BytesIO
from .. import osutils
def inject_bzr_metadata(message, commit_supplement, encoding):
    if not commit_supplement:
        return message
    rt_data = generate_roundtripping_metadata(commit_supplement, encoding)
    if not rt_data:
        return message
    if not isinstance(rt_data, bytes):
        raise TypeError(rt_data)
    return message + b'\n--BZR--\n' + rt_data
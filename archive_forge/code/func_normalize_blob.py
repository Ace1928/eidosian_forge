from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
def normalize_blob(blob, conversion, binary_detection):
    """Takes a blob as input returns either the original blob if
    binary_detection is True and the blob content looks like binary, else
    return a new blob with converted data.
    """
    data = blob.data
    if binary_detection is True:
        if is_binary(data):
            return blob
    converted_data = conversion(data)
    new_blob = Blob()
    new_blob.data = converted_data
    return new_blob
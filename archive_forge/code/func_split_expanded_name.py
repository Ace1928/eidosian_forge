import re
from typing import cast, Dict, Optional, Tuple, MutableMapping, Union
def split_expanded_name(name: str) -> Tuple[str, str]:
    match = EXPANDED_NAME_PATTERN.match(name)
    if match is None:
        raise ValueError('{!r} is not an expanded QName'.format(name))
    namespace, local_name = match.groups()
    return (namespace or '', local_name)
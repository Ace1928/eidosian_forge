import re
from typing import cast, Dict, Optional, Tuple, MutableMapping, Union
def get_prefixed_name(qname: str, namespaces: Union[Dict[str, str], Dict[Optional[str], str]]) -> str:
    """
    Get the prefixed form of a QName, using a namespace map.

    :param qname: an extended QName or a local name or a prefixed QName.
    :param namespaces: a dictionary with a map from prefixes to namespace URIs.
    """
    try:
        if not qname.startswith(('{', 'Q{')):
            return qname
        elif qname[0] == '{':
            ns_uri, local_name = qname[1:].split('}')
        else:
            ns_uri, local_name = qname[2:].split('}')
    except (ValueError, TypeError):
        raise ValueError('{!r} is not a QName'.format(qname))
    for prefix, uri in sorted(namespaces.items(), reverse=True, key=lambda x: x if x[0] is not None else ('', x[1])):
        if uri == ns_uri:
            return '%s:%s' % (prefix, local_name) if prefix else local_name
    else:
        return qname
import re
from typing import cast, Dict, Optional, Tuple, MutableMapping, Union
def get_expanded_name(qname: str, namespaces: Union[Dict[str, str], Dict[Optional[str], str]]) -> str:
    """
    Get the expanded form of a QName, using a namespace map.
    Local names are mapped to the default namespace.

    :param qname: a prefixed QName or a local name or an extended QName.
    :param namespaces: a dictionary with a map from prefixes to namespace URIs.
    :return: the expanded format of a QName or a local name.
    """
    if not qname or qname.startswith('{'):
        return qname
    elif qname.startswith('Q{'):
        return qname[1:]
    try:
        prefix, local_name = qname.split(':')
    except ValueError:
        if ':' in qname:
            raise ValueError('wrong format for prefixed QName %r' % qname)
        elif '' in namespaces:
            uri = namespaces['']
        elif None in namespaces:
            uri = cast(Dict[Optional[str], str], namespaces)[None]
        else:
            return qname
        return '{%s}%s' % (uri, qname) if uri else qname
    else:
        if not prefix or not local_name:
            raise ValueError('wrong format for reference name %r' % qname)
        uri = namespaces[prefix]
        return '{%s}%s' % (uri, local_name) if uri else local_name
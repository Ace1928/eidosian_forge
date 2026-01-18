import xml.etree.ElementTree as ET
def resolve_uri(s, namespaces=cdao_namespaces, cdao_to_obo=True, xml_style=False):
    """Convert prefixed URIs to full URIs.

    Optionally, converts CDAO named identifiers to OBO numeric identifiers.
    """
    if cdao_to_obo and s.startswith('cdao:'):
        return resolve_uri(f'obo:{cdao_elements[s[5:]]}', namespaces, cdao_to_obo)
    for prefix in namespaces:
        if xml_style:
            s = s.replace(prefix + ':', '{%s}' % namespaces[prefix])
        else:
            s = s.replace(prefix + ':', namespaces[prefix])
    return s
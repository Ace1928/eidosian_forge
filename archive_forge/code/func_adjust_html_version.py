from .atom  import atom_add_entry_type
from .html5 import html5_extra_attributes, remove_rel
def adjust_html_version(stream, rdfa_version):
    """
    Adjust the rdfa_version based on the (possible) DTD
    @param stream: the data stream that has to be parsed by an xml parser
    @param rdfa_version: the current rdfa_version; will be returned if nothing else is found
    @return: the rdfa_version, either "1.0" or "1.1, if the DTD says so, otherwise the input rdfa_version value
    """
    import xml.dom.minidom
    parse = xml.dom.minidom.parse
    dom = parse(stream)
    _hl, version = adjust_xhtml_and_version(dom, HostLanguage.xhtml, rdfa_version)
    return version
import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def etree_tostring(elem: ElementProtocol, namespaces: Optional[MutableMapping[str, str]]=None, indent: str='', max_lines: Optional[int]=None, spaces_for_tab: Optional[int]=4, xml_declaration: Optional[bool]=None, encoding: str='unicode', method: str='xml') -> Union[str, bytes]:
    """
    Serialize an Element tree to a string.

    :param elem: the Element instance.
    :param namespaces: is an optional mapping from namespace prefix to URI.     Provided namespaces are registered before serialization. Ignored if the     provided *elem* argument is a lxml Element instance.
    :param indent: the base line indentation.
    :param max_lines: if truncate serialization after a number of lines     (default: do not truncate).
    :param spaces_for_tab: number of spaces for replacing tab characters. For     default tabs are replaced with 4 spaces, provide `None` to keep tab characters.
    :param xml_declaration: if set to `True` inserts the XML declaration at the head.
    :param encoding: if "unicode" (the default) the output is a string,     otherwise it’s binary.
    :param method: is either "xml" (the default), "html" or "text".
    :return: a Unicode string.
    """

    def reindent(line: str) -> str:
        if not line:
            return line
        elif line.startswith(min_indent):
            return line[start:] if start >= 0 else indent[start:] + line
        else:
            return indent + line
    etree_module: Any
    if not is_etree_element(elem):
        raise TypeError(f'{elem!r} is not an Element')
    elif isinstance(elem, PyElementTree.Element):
        etree_module = PyElementTree
    elif not hasattr(elem, 'nsmap'):
        etree_module = ElementTree
    else:
        etree_module = importlib.import_module('lxml.etree')
    if namespaces and (not hasattr(elem, 'nsmap')):
        default_namespace = namespaces.get('')
        for prefix, uri in namespaces.items():
            if prefix and (not re.match('ns\\d+$', prefix)):
                etree_module.register_namespace(prefix, uri)
                if uri == default_namespace:
                    default_namespace = None
        if default_namespace:
            etree_module.register_namespace('', default_namespace)
    xml_text = etree_module.tostring(elem, encoding=encoding, method=method)
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode('utf-8')
    if spaces_for_tab is not None:
        xml_text = xml_text.replace('\t', ' ' * spaces_for_tab)
    if xml_text.startswith('<?xml '):
        if xml_declaration is False:
            lines = xml_text.splitlines()[1:]
        else:
            lines = xml_text.splitlines()
    elif xml_declaration and encoding.lower() != 'unicode':
        lines = ['<?xml version="1.0" encoding="{}"?>'.format(encoding)]
        lines.extend(xml_text.splitlines())
    else:
        lines = xml_text.splitlines()
    while lines and (not lines[-1].strip()):
        lines.pop(-1)
    if not lines or method == 'text' or (not indent and (not max_lines)):
        if encoding == 'unicode':
            return '\n'.join(lines)
        return '\n'.join(lines).encode(encoding)
    last_indent = ' ' * min((k for k in range(len(lines[-1])) if lines[-1][k] != ' '))
    if len(lines) > 2:
        try:
            child_indent = ' ' * min((k for line in lines[1:-1] for k in range(len(line)) if line[k] != ' '))
        except ValueError:
            child_indent = ''
        min_indent = min(child_indent, last_indent)
    else:
        min_indent = child_indent = last_indent
    start = len(min_indent) - len(indent)
    if max_lines is not None and len(lines) > max_lines + 2:
        lines = lines[:max_lines] + [child_indent + '...'] * 2 + lines[-1:]
    if encoding == 'unicode':
        return '\n'.join((reindent(line) for line in lines))
    return '\n'.join((reindent(line) for line in lines)).encode(encoding)
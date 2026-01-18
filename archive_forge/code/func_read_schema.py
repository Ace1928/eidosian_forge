import errno
import getopt
import importlib
import re
import sys
import time
import types
from xml.etree import ElementTree as ElementTree
import saml2
from saml2 import SamlBase
def read_schema(doc, add, defs, impo, modul, ignore, sdir):
    for path in sdir:
        fil = f'{path}{doc}'
        try:
            fp = open(fil)
            fp.close()
            break
        except OSError as e:
            if e.errno == errno.EACCES:
                continue
    else:
        raise Exception('Could not find schema file')
    tree = parse_nsmap(fil)
    known = NAMESPACE_BASE[:]
    known.append(XML_NAMESPACE)
    for key, namespace in tree._root.attrib['xmlns_map'].items():
        if namespace in known:
            continue
        else:
            try:
                modul[key] = modul[namespace]
                impo[namespace][1] = key
            except KeyError:
                if namespace == tree._root.attrib['targetNamespace']:
                    continue
                elif namespace in ignore:
                    continue
                else:
                    raise Exception(f'Undefined namespace: {namespace}')
    _schema = Schema(tree._root, impo, add, modul, defs)
    _included_parts = []
    _remove_parts = []
    _replace = []
    for part in _schema.parts:
        if isinstance(part, Include):
            _sch = read_schema(part.schemaLocation, add, defs, impo, modul, ignore, sdir)
            recursive_add_xmlns_map(_sch, _schema)
            _included_parts.extend(_sch.parts)
            _remove_parts.append(part)
        elif isinstance(part, Redefine):
            _redef = read_schema(part.schemaLocation, add, defs, impo, modul, ignore, sdir)
            _replacement = find_and_replace(_redef, part)
            _replace.append((part, _replacement.parts))
    for part in _remove_parts:
        _schema.parts.remove(part)
    _schema.parts.extend(_included_parts)
    if _replace:
        for vad, med in _replace:
            _schema.parts.remove(vad)
            _schema.parts.extend(med)
    return _schema
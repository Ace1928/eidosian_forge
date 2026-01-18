import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def etree_iter_paths(elem: ElementProtocol, path: str='.') -> Iterator[Tuple[ElementProtocol, str]]:
    yield (elem, path)
    comment_nodes = 0
    pi_nodes = Counter[Optional[str]]()
    positions = Counter[Optional[str]]()
    for child in elem:
        if callable(child.tag):
            if child.tag.__name__ == 'Comment':
                comment_nodes += 1
                yield (child, f'{path}/comment()[{comment_nodes}]')
                continue
            try:
                name = cast(str, child.target)
            except AttributeError:
                assert child.text is not None
                name = child.text.split(' ', maxsplit=1)[0]
            pi_nodes[name] += 1
            yield (child, f'{path}/processing-instruction({name})[{pi_nodes[name]}]')
            continue
        if child.tag.startswith('{'):
            tag = f'Q{child.tag}'
        else:
            tag = f'Q{{}}{child.tag}'
        if path == '/':
            child_path = f'/{tag}'
        elif path:
            child_path = '/'.join((path, tag))
        else:
            child_path = tag
        positions[child.tag] += 1
        child_path += f'[{positions[child.tag]}]'
        yield from etree_iter_paths(child, child_path)
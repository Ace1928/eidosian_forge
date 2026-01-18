from typing import cast, Any, Iterator, List, Optional, Union
from .namespaces import NamespacesType
from .exceptions import ElementPathTypeError
from .protocols import ElementProtocol, LxmlElementProtocol, \
from .etree import is_etree_document, is_etree_element
from .xpath_nodes import SchemaElemType, ChildNodeType, ElementMapType, \
def build_lxml_node_tree(root: LxmlRootType, uri: Optional[str]=None, fragment: bool=False) -> Union[DocumentNode, ElementNode]:
    """
    Returns a tree of XPath nodes that wrap the provided lxml root tree.

    :param root: a lxml Element or a lxml ElementTree.
    :param uri: an optional URI associated with the document or the root element.
    :param fragment: if `True` a root element is considered a fragment, otherwise     a root element is considered the root of an XML document.
    """
    root_node: Union[DocumentNode, ElementNode]
    parent: Any
    elements: Any
    child: ChildNodeType
    children: Iterator[Any]
    position = 1

    def build_lxml_element_node() -> ElementNode:
        nonlocal position
        node = ElementNode(elem, parent, position, elem.nsmap)
        position += 1
        elements[elem] = node
        position += len(elem.nsmap) + int('xml' not in elem.nsmap) + len(elem.attrib)
        if elem.text is not None:
            node.children.append(TextNode(elem.text, node, position))
            position += 1
        return node

    def build_document_node() -> ElementNode:
        nonlocal position
        nonlocal child
        for e in reversed([x for x in elem.itersiblings(preceding=True)]):
            if e.tag.__name__ == 'Comment':
                parent.children.append(CommentNode(e, parent, position))
            else:
                parent.children.append(ProcessingInstructionNode(e, parent, position))
            position += 1
        node = build_lxml_element_node()
        parent.children.append(node)
        for e in elem.itersiblings():
            if e.tag.__name__ == 'Comment':
                parent.children.append(CommentNode(e, parent, position))
            else:
                parent.children.append(ProcessingInstructionNode(e, parent, position))
            position += 1
        return node
    if fragment:
        if hasattr(root, 'parse'):
            root_elem = cast(LxmlDocumentProtocol, root).getroot()
            if root_elem is None:
                msg = 'requested a fragment of an empty ElementTree document'
                raise ElementPathTypeError(msg)
            elem = root_elem
        else:
            elem = root
        parent = None
        elements = {}
        root_node = parent = build_lxml_element_node()
        root_node.elements = elements
    elif hasattr(root, 'parse'):
        document = cast(LxmlDocumentProtocol, root)
        root_node = parent = DocumentNode(document, position=position)
        position += 1
        root_elem = document.getroot()
        if root_elem is None:
            return root_node
        elem = root_elem
        elements = root_node.elements
        parent = build_document_node()
    elif root.getparent() is None:
        document = root.getroottree()
        root_node = parent = DocumentNode(document, position=0)
        root_elem = document.getroot()
        assert root_elem is not None
        elem = root_elem
        elements = root_node.elements
        parent = build_document_node()
        if len(root_node.children) == 1:
            parent.elements = root_node.elements
            parent.parent = None
            root_node = parent
    else:
        elem = root
        parent = None
        elements = {}
        root_node = parent = build_lxml_element_node()
        root_node.elements = elements
    children = iter(elem)
    iterators: List[Any] = []
    ancestors: List[Any] = []
    if uri is not None:
        root_node.uri = uri
    while True:
        for elem in children:
            if not callable(elem.tag):
                child = build_lxml_element_node()
            elif elem.tag.__name__ == 'Comment':
                child = CommentNode(elem, parent, position)
                position += 1
            else:
                child = ProcessingInstructionNode(elem, parent, position)
            parent.children.append(child)
            if elem.tail is not None:
                parent.children.append(TextNode(elem.tail, parent, position))
                position += 1
            if len(elem):
                ancestors.append(parent)
                parent = child
                iterators.append(children)
                children = iter(elem)
                break
        else:
            try:
                children, parent = (iterators.pop(), ancestors.pop())
            except IndexError:
                return root_node
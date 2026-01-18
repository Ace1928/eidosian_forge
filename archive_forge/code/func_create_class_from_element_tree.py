import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def create_class_from_element_tree(target_class, tree, namespace=None, tag=None):
    """Instantiates the class and populates members according to the tree.

    Note: Only use this function with classes that have c_namespace and c_tag
    class members.

    :param target_class: The class which will be instantiated and populated
        with the contents of the XML.
    :param tree: An element tree whose contents will be converted into
        members of the new target_class instance.
    :param namespace: The namespace which the XML tree's root node must
        match. If omitted, the namespace defaults to the c_namespace of the
        target class.
    :param tag: The tag which the XML tree's root node must match. If
        omitted, the tag defaults to the c_tag class member of the target
        class.

    :return: An instance of the target class - or None if the tag and namespace
        of the XML tree's root node did not match the desired namespace and tag.
    """
    if namespace is None:
        namespace = target_class.c_namespace
    if tag is None:
        tag = target_class.c_tag
    if tree.tag == f'{{{namespace}}}{tag}':
        target = target_class()
        target.harvest_element_tree(tree)
        return target
    else:
        return None
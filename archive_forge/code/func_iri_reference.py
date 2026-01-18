from .iri import IRIReference
from .parseresult import ParseResult
from .uri import URIReference
def iri_reference(iri, encoding='utf-8'):
    """Parse a IRI string into an IRIReference.

    This is a convenience function. You could achieve the same end by using
    ``IRIReference.from_string(iri)``.

    :param str iri: The IRI which needs to be parsed into a reference.
    :param str encoding: The encoding of the string provided
    :returns: A parsed IRI
    :rtype: :class:`IRIReference`
    """
    return IRIReference.from_string(iri, encoding)
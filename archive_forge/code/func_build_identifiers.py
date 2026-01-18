import jmespath
from botocore import xform_name
from .params import get_data_member
def build_identifiers(identifiers, parent, params=None, raw_response=None):
    """
    Builds a mapping of identifier names to values based on the
    identifier source location, type, and target. Identifier
    values may be scalars or lists depending on the source type
    and location.

    :type identifiers: list
    :param identifiers: List of :py:class:`~boto3.resources.model.Parameter`
                        definitions
    :type parent: ServiceResource
    :param parent: The resource instance to which this action is attached.
    :type params: dict
    :param params: Request parameters sent to the service.
    :type raw_response: dict
    :param raw_response: Low-level operation response.
    :rtype: list
    :return: An ordered list of ``(name, value)`` identifier tuples.
    """
    results = []
    for identifier in identifiers:
        source = identifier.source
        target = identifier.target
        if source == 'response':
            value = jmespath.search(identifier.path, raw_response)
        elif source == 'requestParameter':
            value = jmespath.search(identifier.path, params)
        elif source == 'identifier':
            value = getattr(parent, xform_name(identifier.name))
        elif source == 'data':
            value = get_data_member(parent, identifier.path)
        elif source == 'input':
            continue
        else:
            raise NotImplementedError('Unsupported source type: {0}'.format(source))
        results.append((xform_name(target), value))
    return results
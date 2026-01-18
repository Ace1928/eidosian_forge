from __future__ import absolute_import
import logging
import os
def get_static_doc(serviceName, version):
    """Retrieves the discovery document from the directory defined in
    DISCOVERY_DOC_DIR corresponding to the serviceName and version provided.

    Args:
        serviceName: string, name of the service.
        version: string, the version of the service.

    Returns:
        A string containing the contents of the JSON discovery document,
        otherwise None if the JSON discovery document was not found.
    """
    content = None
    doc_name = '{}.{}.json'.format(serviceName, version)
    try:
        with open(os.path.join(DISCOVERY_DOC_DIR, doc_name), 'r') as f:
            content = f.read()
    except FileNotFoundError:
        pass
    return content
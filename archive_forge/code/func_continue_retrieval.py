import logging
from oslo_utils import timeutils
from suds import sudsobject
def continue_retrieval(vim, retrieve_result):
    """Continue retrieving results, if available.

    :param vim: Vim object
    :param retrieve_result: result of RetrievePropertiesEx API call
    :raises: VimException, VimFaultException, VimAttributeException,
             VimSessionOverLoadException, VimConnectionException
    """
    token = _get_token(retrieve_result)
    if token:
        collector = vim.service_content.propertyCollector
        return vim.ContinueRetrievePropertiesEx(collector, token=token)
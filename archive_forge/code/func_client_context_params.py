from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.client import (
from botocore.docs.paginator import PaginatorDocumenter
from botocore.docs.waiter import WaiterDocumenter
from botocore.exceptions import DataNotFoundError
def client_context_params(self, section):
    omitted_params = ClientContextParamsDocumenter.OMITTED_CONTEXT_PARAMS
    params_to_omit = omitted_params.get(self._service_name, [])
    service_model = self._client.meta.service_model
    raw_context_params = service_model.client_context_parameters
    context_params = [p for p in raw_context_params if p.name not in params_to_omit]
    if context_params:
        context_param_documenter = ClientContextParamsDocumenter(self._service_name, context_params)
        context_param_documenter.document_context_params(section)
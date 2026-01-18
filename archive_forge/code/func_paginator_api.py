from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.client import (
from botocore.docs.paginator import PaginatorDocumenter
from botocore.docs.waiter import WaiterDocumenter
from botocore.exceptions import DataNotFoundError
def paginator_api(self, section):
    try:
        service_paginator_model = self._session.get_paginator_model(self._service_name)
    except DataNotFoundError:
        return
    if service_paginator_model._paginator_config:
        paginator_documenter = PaginatorDocumenter(self._client, service_paginator_model, self._root_docs_path)
        paginator_documenter.document_paginators(section)
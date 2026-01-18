import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
def document_paginators(self, section):
    """Documents the various paginators for a service

        param section: The section to write to.
        """
    section.style.h2('Paginators')
    self._add_overview(section)
    section.style.new_line()
    section.writeln('The available paginators are:')
    section.style.toctree()
    paginator_names = sorted(self._service_paginator_model._paginator_config)
    for paginator_name in paginator_names:
        section.style.tocitem(f'{self._service_name}/paginator/{paginator_name}')
        paginator_doc_structure = DocumentStructure(paginator_name, target='html')
        self._add_paginator(paginator_doc_structure, paginator_name)
        paginator_dir_path = os.path.join(self._root_docs_path, self._service_name, 'paginator')
        paginator_doc_structure.write_to_file(paginator_dir_path, paginator_name)
import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
def document_waiters(self, section):
    """Documents the various waiters for a service.

        :param section: The section to write to.
        """
    section.style.h2('Waiters')
    self._add_overview(section)
    section.style.new_line()
    section.writeln('The available waiters are:')
    section.style.toctree()
    for waiter_name in self._service_waiter_model.waiter_names:
        section.style.tocitem(f'{self._service_name}/waiter/{waiter_name}')
        waiter_doc_structure = DocumentStructure(waiter_name, target='html')
        self._add_single_waiter(waiter_doc_structure, waiter_name)
        waiter_dir_path = os.path.join(self._root_docs_path, self._service_name, 'waiter')
        waiter_doc_structure.write_to_file(waiter_dir_path, waiter_name)
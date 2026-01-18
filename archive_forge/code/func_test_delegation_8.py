from prov.model import *
def test_delegation_8(self):
    document = self.new_document()
    dele = document.delegation(EX_NS['e1'], EX_NS['ag1'], activity=EX_NS['a1'], identifier=EX_NS['dele8'])
    self.add_labels(dele)
    self.add_types(dele)
    self.add_further_attributes(dele)
    self.do_tests(document)
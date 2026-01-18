from prov.model import *
def test_delegation_6(self):
    document = self.new_document()
    dele = document.delegation(EX_NS['e1'], EX_NS['ag1'], activity=EX_NS['a1'], identifier=EX_NS['dele6'])
    self.add_labels(dele)
    self.do_tests(document)
from prov.model import *
def test_derivation_4(self):
    document = self.new_document()
    der = document.derivation(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['der4'])
    self.add_label(der)
    self.do_tests(document)
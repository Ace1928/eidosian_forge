from prov.model import *
def test_communication_7(self):
    document = self.new_document()
    inf = document.communication(EX_NS['a2'], EX_NS['a1'], identifier=EX_NS['inf7'])
    self.add_labels(inf)
    self.add_types(inf)
    self.add_further_attributes(inf)
    self.do_tests(document)
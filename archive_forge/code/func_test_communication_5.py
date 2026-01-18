from prov.model import *
def test_communication_5(self):
    document = self.new_document()
    inf = document.communication(EX_NS['a2'], EX_NS['a1'], identifier=EX_NS['inf5'])
    self.add_labels(inf)
    self.do_tests(document)
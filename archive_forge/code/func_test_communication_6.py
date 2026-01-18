from prov.model import *
def test_communication_6(self):
    document = self.new_document()
    inf = document.communication(EX_NS['a2'], EX_NS['a1'], identifier=EX_NS['inf6'])
    self.add_labels(inf)
    self.add_types(inf)
    self.do_tests(document)
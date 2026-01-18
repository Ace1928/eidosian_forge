from prov.model import *
def test_communication_2(self):
    document = self.new_document()
    document.communication(None, EX_NS['a1'], identifier=EX_NS['inf2'])
    self.do_tests(document)
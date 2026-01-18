from prov.model import *
def test_communication_3(self):
    document = self.new_document()
    document.communication(EX_NS['a2'], EX_NS['a1'], identifier=EX_NS['inf3'])
    self.do_tests(document)
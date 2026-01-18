from prov.model import *
def test_communication_4(self):
    document = self.new_document()
    document.communication(EX_NS['a2'], EX_NS['a1'])
    self.do_tests(document)
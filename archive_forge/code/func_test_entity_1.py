from prov.model import *
def test_entity_1(self):
    document = self.new_document()
    document.entity(EX_NS['e1'])
    self.do_tests(document)
from prov.model import *
def test_alternate_1(self):
    document = self.new_document()
    document.alternate(EX_NS['e2'], EX_NS['e1'])
    self.do_tests(document)
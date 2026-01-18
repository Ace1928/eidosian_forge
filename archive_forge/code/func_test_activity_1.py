from prov.model import *
def test_activity_1(self):
    document = self.new_document()
    document.activity(EX_NS['a1'])
    self.do_tests(document)
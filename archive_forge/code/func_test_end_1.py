from prov.model import *
def test_end_1(self):
    document = self.new_document()
    document.end(None, trigger=EX_NS['e1'], identifier=EX_NS['end1'])
    self.do_tests(document)
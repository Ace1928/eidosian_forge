from prov.model import *
def test_end_5(self):
    document = self.new_document()
    document.end(EX_NS['a1'], trigger=EX_NS['e1'], identifier=EX_NS['end5'], ender=EX_NS['a2'])
    self.do_tests(document)
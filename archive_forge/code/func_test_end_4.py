from prov.model import *
def test_end_4(self):
    document = self.new_document()
    document.end(None, trigger=EX_NS['e1'], identifier=EX_NS['end4'], ender=EX_NS['a2'])
    self.do_tests(document)
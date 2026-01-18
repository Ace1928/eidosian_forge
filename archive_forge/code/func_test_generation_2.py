from prov.model import *
def test_generation_2(self):
    document = self.new_document()
    document.generation(EX_NS['e1'], identifier=EX_NS['gen2'], activity=EX_NS['a1'])
    self.do_tests(document)
from prov.model import *
def test_generation_3(self):
    document = self.new_document()
    a = document.generation(EX_NS['e1'], identifier=EX_NS['gen3'], activity=EX_NS['a1'])
    a.add_attributes([(PROV_ROLE, 'somerole'), (PROV_ROLE, 'otherRole')])
    self.do_tests(document)
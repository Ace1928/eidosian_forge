import codecs
from nltk.sem import evaluate
def demo_model0():
    global m0, g0
    v = [('john', 'b1'), ('mary', 'g1'), ('suzie', 'g2'), ('fido', 'd1'), ('tess', 'd2'), ('noosa', 'n'), ('girl', {'g1', 'g2'}), ('boy', {'b1', 'b2'}), ('dog', {'d1', 'd2'}), ('bark', {'d1', 'd2'}), ('walk', {'b1', 'g2', 'd1'}), ('chase', {('b1', 'g1'), ('b2', 'g1'), ('g1', 'd1'), ('g2', 'd2')}), ('see', {('b1', 'g1'), ('b2', 'd2'), ('g1', 'b1'), ('d2', 'b1'), ('g2', 'n')}), ('in', {('b1', 'n'), ('b2', 'n'), ('d2', 'n')}), ('with', {('b1', 'g1'), ('g1', 'b1'), ('d1', 'b1'), ('b1', 'd1')})]
    val = evaluate.Valuation(v)
    dom = val.domain
    m0 = evaluate.Model(dom, val)
    g0 = evaluate.Assignment(dom)
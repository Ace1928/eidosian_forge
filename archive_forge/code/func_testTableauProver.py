from nltk.inference.api import BaseProverCommand, Prover
from nltk.internals import Counter
from nltk.sem.logic import (
def testTableauProver():
    tableau_test('P | -P')
    tableau_test('P & -P')
    tableau_test('Q', ['P', '(P -> Q)'])
    tableau_test('man(x)')
    tableau_test('(man(x) -> man(x))')
    tableau_test('(man(x) -> --man(x))')
    tableau_test('-(man(x) and -man(x))')
    tableau_test('(man(x) or -man(x))')
    tableau_test('(man(x) -> man(x))')
    tableau_test('-(man(x) and -man(x))')
    tableau_test('(man(x) or -man(x))')
    tableau_test('(man(x) -> man(x))')
    tableau_test('(man(x) iff man(x))')
    tableau_test('-(man(x) iff -man(x))')
    tableau_test('all x.man(x)')
    tableau_test('all x.all y.((x = y) -> (y = x))')
    tableau_test('all x.all y.all z.(((x = y) & (y = z)) -> (x = z))')
    p1 = 'all x.(man(x) -> mortal(x))'
    p2 = 'man(Socrates)'
    c = 'mortal(Socrates)'
    tableau_test(c, [p1, p2])
    p1 = 'all x.(man(x) -> walks(x))'
    p2 = 'man(John)'
    c = 'some y.walks(y)'
    tableau_test(c, [p1, p2])
    p = '((x = y) & walks(y))'
    c = 'walks(x)'
    tableau_test(c, [p])
    p = '((x = y) & ((y = z) & (z = w)))'
    c = '(x = w)'
    tableau_test(c, [p])
    p = 'some e1.some e2.(believe(e1,john,e2) & walk(e2,mary))'
    c = 'some e0.walk(e0,mary)'
    tableau_test(c, [p])
    c = '(exists x.exists z3.((x = Mary) & ((z3 = John) & sees(z3,x))) <-> exists x.exists z4.((x = John) & ((z4 = Mary) & sees(x,z4))))'
    tableau_test(c)
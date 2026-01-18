import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_hyperhyponyms(self):
    self.assertEqual(S('travel.v.01').hypernyms(), [])
    self.assertEqual(S('travel.v.02').hypernyms(), [S('travel.v.03')])
    self.assertEqual(S('travel.v.03').hypernyms(), [])
    self.assertEqual(S('breakfast.n.1').hypernyms(), [S('meal.n.01')])
    first_five_meal_hypo = [S('banquet.n.02'), S('bite.n.04'), S('breakfast.n.01'), S('brunch.n.01'), S('buffet.n.02')]
    self.assertEqual(sorted(S('meal.n.1').hyponyms()[:5]), first_five_meal_hypo)
    self.assertEqual(S('Austen.n.1').instance_hypernyms(), [S('writer.n.01')])
    first_five_composer_hypo = [S('ambrose.n.01'), S('bach.n.01'), S('barber.n.01'), S('bartok.n.01'), S('beethoven.n.01')]
    self.assertEqual(S('composer.n.1').instance_hyponyms()[:5], first_five_composer_hypo)
    self.assertEqual(S('person.n.01').root_hypernyms(), [S('entity.n.01')])
    self.assertEqual(S('sail.v.01').root_hypernyms(), [S('travel.v.01')])
    self.assertEqual(S('fall.v.12').root_hypernyms(), [S('act.v.01'), S('fall.v.17')])
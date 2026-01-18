import unittest
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wnic
def test_meronyms_holonyms(self):
    self.assertEqual(S('dog.n.01').member_holonyms(), [S('canis.n.01'), S('pack.n.06')])
    self.assertEqual(S('dog.n.01').part_meronyms(), [S('flag.n.07')])
    self.assertEqual(S('faculty.n.2').member_meronyms(), [S('professor.n.01')])
    self.assertEqual(S('copilot.n.1').member_holonyms(), [S('crew.n.01')])
    self.assertEqual(S('table.n.2').part_meronyms(), [S('leg.n.03'), S('tabletop.n.01'), S('tableware.n.01')])
    self.assertEqual(S('course.n.7').part_holonyms(), [S('meal.n.01')])
    self.assertEqual(S('water.n.1').substance_meronyms(), [S('hydrogen.n.01'), S('oxygen.n.01')])
    self.assertEqual(S('gin.n.1').substance_holonyms(), [S('gin_and_it.n.01'), S('gin_and_tonic.n.01'), S('martini.n.01'), S('pink_lady.n.01')])
import unittest
from nltk.corpus import brown
from nltk.jsontags import JSONTaggedDecoder, JSONTaggedEncoder
from nltk.tag import (
from nltk.tag.brill import nltkdemo18
def test_affix_tagger(self):
    tagger = AffixTagger(self.corpus, backoff=self.default_tagger)
    encoded = self.encoder.encode(tagger)
    decoded = self.decoder.decode(encoded)
    self.assertEqual(repr(tagger), repr(decoded))
    self.assertEqual(repr(tagger.backoff), repr(decoded.backoff))
    self.assertEqual(tagger._affix_length, decoded._affix_length)
    self.assertEqual(tagger._min_word_length, decoded._min_word_length)
    self.assertEqual(tagger._context_to_tag, decoded._context_to_tag)
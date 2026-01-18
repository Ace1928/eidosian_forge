import unittest
from nltk.corpus import brown
from nltk.jsontags import JSONTaggedDecoder, JSONTaggedEncoder
from nltk.tag import (
from nltk.tag.brill import nltkdemo18
def test_ngram_taggers(self):
    unitagger = UnigramTagger(self.corpus, backoff=self.default_tagger)
    bitagger = BigramTagger(self.corpus, backoff=unitagger)
    tritagger = TrigramTagger(self.corpus, backoff=bitagger)
    ntagger = NgramTagger(4, self.corpus, backoff=tritagger)
    encoded = self.encoder.encode(ntagger)
    decoded = self.decoder.decode(encoded)
    self.assertEqual(repr(ntagger), repr(decoded))
    self.assertEqual(repr(tritagger), repr(decoded.backoff))
    self.assertEqual(repr(bitagger), repr(decoded.backoff.backoff))
    self.assertEqual(repr(unitagger), repr(decoded.backoff.backoff.backoff))
    self.assertEqual(repr(self.default_tagger), repr(decoded.backoff.backoff.backoff.backoff))
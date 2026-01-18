import re
from warnings import warn
from xml.etree import ElementTree as et
from nltk.corpus.reader import CorpusReader
def load_wiki_q(self):
    """Load conversion table to Wikidata Q-codes (only if needed)"""
    with self.open('cldr/tools-cldr-rdf-external-entityToCode.tsv') as fp:
        self.wiki_q = self.wiki_dict(fp.read().strip().split('\n')[1:])
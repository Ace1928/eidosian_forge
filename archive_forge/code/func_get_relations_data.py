import base64
import copy
import getopt
import io
import os
import pickle
import sys
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from sys import argv
from urllib.parse import unquote_plus
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
def get_relations_data(word, synset):
    """
    Get synset relations data for a synset.  Note that this doesn't
    yet support things such as full hyponym vs direct hyponym.
    """
    if synset.pos() == wn.NOUN:
        return ((HYPONYM, 'Hyponyms', synset.hyponyms()), (INSTANCE_HYPONYM, 'Instance hyponyms', synset.instance_hyponyms()), (HYPERNYM, 'Direct hypernyms', synset.hypernyms()), (INDIRECT_HYPERNYMS, 'Indirect hypernyms', rebuild_tree(synset.tree(lambda x: x.hypernyms()))[1]), (INSTANCE_HYPERNYM, 'Instance hypernyms', synset.instance_hypernyms()), (PART_HOLONYM, 'Part holonyms', synset.part_holonyms()), (PART_MERONYM, 'Part meronyms', synset.part_meronyms()), (SUBSTANCE_HOLONYM, 'Substance holonyms', synset.substance_holonyms()), (SUBSTANCE_MERONYM, 'Substance meronyms', synset.substance_meronyms()), (MEMBER_HOLONYM, 'Member holonyms', synset.member_holonyms()), (MEMBER_MERONYM, 'Member meronyms', synset.member_meronyms()), (ATTRIBUTE, 'Attributes', synset.attributes()), (ANTONYM, 'Antonyms', lemma_property(word, synset, lambda l: l.antonyms())), (DERIVATIONALLY_RELATED_FORM, 'Derivationally related form', lemma_property(word, synset, lambda l: l.derivationally_related_forms())))
    elif synset.pos() == wn.VERB:
        return ((ANTONYM, 'Antonym', lemma_property(word, synset, lambda l: l.antonyms())), (HYPONYM, 'Hyponym', synset.hyponyms()), (HYPERNYM, 'Direct hypernyms', synset.hypernyms()), (INDIRECT_HYPERNYMS, 'Indirect hypernyms', rebuild_tree(synset.tree(lambda x: x.hypernyms()))[1]), (ENTAILMENT, 'Entailments', synset.entailments()), (CAUSE, 'Causes', synset.causes()), (ALSO_SEE, 'Also see', synset.also_sees()), (VERB_GROUP, 'Verb Groups', synset.verb_groups()), (DERIVATIONALLY_RELATED_FORM, 'Derivationally related form', lemma_property(word, synset, lambda l: l.derivationally_related_forms())))
    elif synset.pos() == wn.ADJ or synset.pos == wn.ADJ_SAT:
        return ((ANTONYM, 'Antonym', lemma_property(word, synset, lambda l: l.antonyms())), (SIMILAR, 'Similar to', synset.similar_tos()), (PERTAINYM, 'Pertainyms', lemma_property(word, synset, lambda l: l.pertainyms())), (ATTRIBUTE, 'Attributes', synset.attributes()), (ALSO_SEE, 'Also see', synset.also_sees()))
    elif synset.pos() == wn.ADV:
        return ((ANTONYM, 'Antonym', lemma_property(word, synset, lambda l: l.antonyms())),)
    else:
        raise TypeError('Unhandles synset POS type: ' + str(synset.pos()))
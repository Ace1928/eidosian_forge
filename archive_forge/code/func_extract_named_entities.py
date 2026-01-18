import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk, tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
def extract_named_entities(tagged_sentence: List[tuple]) -> List[Topic]:
    """
    Extract named entities from the tagged sentence, categorize, score, and contextualize them.
    """
    chunked = ne_chunk(tagged_sentence)
    iob_tagged = tree2conlltags(chunked)
    topics = defaultdict(list)
    word_freq = Counter([word for word, tag in tagged_sentence])
    for index, (word, tag, iob) in enumerate(iob_tagged):
        if iob != 'O':
            entity_type = iob.split('-')[1]
            topics[entity_type].append((word, index))
    processed_topics = []
    for category, entities in topics.items():
        for entity, position in entities:
            significance = word_freq[entity] * semantic_similarity(entity, category)
            context_window = tagged_sentence[max(0, position - 2):min(len(tagged_sentence), position + 3)]
            associated_phrases = [word for word, _ in context_window]
            processed_topics.append(Topic(entity, category, significance, position, associated_phrases))
    return processed_topics
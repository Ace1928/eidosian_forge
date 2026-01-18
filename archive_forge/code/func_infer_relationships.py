import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk, tree2conlltags
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
def infer_relationships(tagged_sentence: List[tuple], topics: List[Topic], full_text: str) -> List[Topic]:
    """
    Infer relationships between topics based on their context and position in the full text.
    """
    for topic in topics:
        for other_topic in topics:
            if topic != other_topic:
                if any((word in full_text for word in topic.associated_phrases)) and any((word in full_text for word in other_topic.associated_phrases)):
                    topic.relationships.append((other_topic.name, 'thematic connection'))
    return topics